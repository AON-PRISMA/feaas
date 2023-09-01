package state_machine

import (
	"Server/infectious"
	"fmt"
	socketio "github.com/googollee/go-socket.io"
	"log"
	"strconv"
	"sync"
	"time"
)

func sendMessage(clientsId []int, server *socketio.Server, msg Message, event string) {
	for _, r := range clientsId {
		server.BroadcastToRoom("/", strconv.Itoa(r), event, msg)
	}
}

func rangeList(lo int, hi int) []int {
	clientList := make([]int, hi-lo+1)
	for i := range clientList {
		clientList[i] = i + lo
	}
	return clientList
}

func bw(r1 []infectious.MyInt, r2 []infectious.MyInt, retVal *bool, rep, th int, plain int) {
	if len(r1) != len(r2) {
		log.Fatalf("Non matching encoding length (%d, %d)", len(r1), len(r2))
	}
	dim := len(r1)
	if plain != 0 {
		*retVal = infectious.MatchRecordsPlain(r1, r2, dim, rep, th)
	} else {
		*retVal = infectious.MatchRecords(r1, r2, dim, rep, th)
	}
}

func fetchRecord(db *Database_, cId int, idx int) (*Record, bool) {
	db.RLock()
	defer db.RUnlock()
	if idx >= len(db.Table[cId]) {
		return nil, false
	} else {
		return db.Table[cId][idx], true
	}
}

// matching function without LSH
func matcher(s *ServerStateMachine) {
	k := 0
	if _, ok := s.MatchingResults[s.CurClientId]; !ok {
		var totalResults [][][]bool
		s.MatchingResults[s.CurClientId] = totalResults
	}
	for {
		curClientRecord, found := fetchRecord(s.Database, s.CurClientId, k) // fetch a single record
		if !found {
			time.Sleep(100 * time.Millisecond)
			continue
		}
		fmt.Printf("Performing matching for record id %d...\n", curClientRecord.RecordId+1)
		var wg sync.WaitGroup
		// prepare a 2d array
		results := make([][]bool, s.CurClientId-1) // len >= 1
		s.Database.RLock()
		for i := range results {
			results[i] = make([]bool, len(s.Database.Table[i+1]))
		}
		s.Database.RUnlock()
		//fmt.Println("Cur record idx", k)
		start := time.Now()
		count := 0
		for i := 1; i <= s.CurClientId-1; i++ {
			s.Database.RLock()
			for j, r2 := range s.Database.Table[i] {
				//fmt.Println("Matching client id:", i, "Matching", j)
				count += 1
				s.NumMatchCount += 1
				wg.Add(1)
				// shadows loop vars
				i, j, r2 := i, j, r2
				go func() {
					defer wg.Done()
					bw(curClientRecord.DataRecord, r2.DataRecord, &results[i-1][j], s.rep, s.Threshold, s.plain)
				}()
			}
			s.Database.RUnlock()
		}
		wg.Wait()
		elapsed := time.Since(start)
		if count > 0 {
			log.Printf("Average time for a matching: %s (%d matches)", elapsed/time.Duration(count), count)
		} else {
			log.Printf("Average time for a matching: 0")
		}

		//fmt.Println("Results:")
		//fmt.Println(results)
		s.MatchingResults[s.CurClientId] = append(s.MatchingResults[s.CurClientId], results)
		// check match exists?
		for i := 0; i < len(results); i++ {
			for j := 0; j < len(results[0]); j++ {
				if results[i][j] {
					if curClientRecord.LastRecord {
						s.ServerStates.MatchOccursLastRecord = true
					}
					s.ServerStates.MatchOccurs = true
					s.ServerStates.MatchingDone = true
					return
				}
			}
		}
		if curClientRecord.LastRecord {
			s.ServerStates.MatchingDone = true
			return
		}
		k += 1
	}
}

// matching function with LSH
func matcherLSH(s *ServerStateMachine) {
	k := 0
	if _, ok := s.MatchingResults[s.CurClientId]; !ok {
		var totalResults [][][]bool
		s.MatchingResults[s.CurClientId] = totalResults
	}
	for {
		curClientRecord, found := fetchRecord(s.Database, s.CurClientId, k)
		if !found {
			time.Sleep(100 * time.Millisecond)
			continue
		}
		fmt.Printf("Performing matching for record id %d...\n", curClientRecord.RecordId+1)
		// prepare a 2d array
		s.Database.RLock()
		results := make([][]bool, s.CurClientId-1) // len >= 1
		for i := range results {
			results[i] = make([]bool, len(s.Database.Table[i+1]))
		}

		// this 2d array indicates which records have been compared already
		hasResults := make([][]bool, s.CurClientId-1) // len >= 1
		for i := range hasResults {
			hasResults[i] = make([]bool, len(s.Database.Table[i+1]))
		}
		s.Database.RUnlock()

		var wg sync.WaitGroup
		start := time.Now()
		count := 0
		// loop through range(lshRep)
		for i := 0; i < len(s.Database.LSHTable); i++ {
			s.Database.RLock()
			// get all record info from the corresponding bin
			possibleMatches := s.Database.LSHTable[i][curClientRecord.LSHVal[i]]
			for j := 0; j < len(possibleMatches); j++ {
				clientIndex, recordIndex := possibleMatches[j].ClientIndex, possibleMatches[j].RecordIndex
				if clientIndex >= s.CurClientId { // ignore record with larger id
					continue
				}
				if hasResults[clientIndex-1][recordIndex] { // check if this record has been compared before
					continue
				}
				count += 1
				s.NumMatchCount += 1
				wg.Add(1)
				go func() {
					defer wg.Done()
					bw(curClientRecord.DataRecord, s.Database.Table[clientIndex][recordIndex].DataRecord,
						&results[clientIndex-1][recordIndex], s.rep, s.Threshold, s.plain)
				}()
				hasResults[clientIndex-1][recordIndex] = true
			}
			s.Database.RUnlock()
		}
		wg.Wait()
		elapsed := time.Since(start)
		if count > 0 {
			log.Printf("Average time for a matching: %s (%d matches)", elapsed/time.Duration(count), count)
		} else {
			log.Printf("Average time for a matching: 0")
		}

		//fmt.Println("Results:")
		//fmt.Println(results)
		s.MatchingResults[s.CurClientId] = append(s.MatchingResults[s.CurClientId], results)
		// check match exists?
		for i := 0; i < len(results); i++ {
			for j := 0; j < len(results[0]); j++ {
				if results[i][j] {
					if curClientRecord.LastRecord {
						s.ServerStates.MatchOccursLastRecord = true
					}
					s.ServerStates.MatchOccurs = true
					s.ServerStates.MatchingDone = true
					return
				}
			}
		}
		if curClientRecord.LastRecord {
			s.ServerStates.MatchingDone = true
			return
		}
		k += 1
	}
}

// Get the matching info for the current client
func getMatchingIds(s *ServerStateMachine) []MatchingInfo {
	var ret []MatchingInfo
	// only look at the last row of the 3d array
	matchingTable := s.MatchingResults[s.CurClientId][len(s.MatchingResults[s.CurClientId])-1]
	for i := 0; i < len(matchingTable); i++ {
		for j := 0; j < len(matchingTable[0]); j++ {
			if matchingTable[i][j] {
				ret = append(ret, MatchingInfo{
					StreamingClientId:  s.CurClientId,
					StreamingClientInd: len(s.MatchingResults[s.CurClientId]) - 1,
					MatchingClientId:   i + 1,
					MatchingClientInd:  j,
				})
			}
		}
	}
	return ret
}
