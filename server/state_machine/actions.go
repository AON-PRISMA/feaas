package state_machine

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"sort"
	"strconv"
	"time"
)

var startTime time.Time

type S0Action struct{}
type S1Action struct{}
type S2Action struct{}
type S3Action struct{}
type S3aAction struct{}
type S4Action struct{}
type S5Action struct{}
type S5aAction struct{}
type S6Action struct{}
type S6aAction struct{}
type S7Action struct{}

func (a *S0Action) Execute(_ *ServerStateMachine) {
	return
}

func (a *S1Action) Execute(s *ServerStateMachine) {
	startTime = time.Now()
	// clear m1 flag to prevent infinite loop
	s.ClientsMsgStatus.Lock()
	defer s.ClientsMsgStatus.Unlock()
	s.ClientsMsgStatus.ReceivedM1 = map[int]bool{}

	s.CurClientId = 1
	for i := 1; i <= s.TotalNumClients; i++ {
		msg := Message{
			Id:      2,
			Desc:    "Initiate key exchange",
			Content: map[string]any{"clientId": i},
		}
		sendMessage([]int{i}, s.Server, msg, "message")
	}
	return
}

func (a *S2Action) Execute(_ *ServerStateMachine) {
	return
}

func (a *S3Action) Execute(s *ServerStateMachine) {
	s.CurClientId += 1

	// clear m5 flag
	s.ClientsMsgStatus.Lock()
	s.ClientsMsgStatus.ReceivedM5 = map[int]bool{}
	s.ClientsMsgStatus.Unlock()

	// tell the other clients to resend all records.
	msg := Message{
		Id:      8,
		Desc:    "Send records",
		Content: map[string]any{"sendAll": true},
	}

	clientList := rangeList(1, s.CurClientId-1)
	sendMessage(clientList, s.Server, msg, "message")

	// tell the streaming clients not to send previous records.
	msg = Message{
		Id:      8,
		Desc:    "Send records",
		Content: map[string]any{"sendAll": false},
	}

	sendMessage([]int{s.CurClientId}, s.Server, msg, "message")

	return
}

func (a *S3aAction) Execute(s *ServerStateMachine) {
	s.CurClientId += 1

	// clear m5 flag
	s.ClientsMsgStatus.Lock()
	s.ClientsMsgStatus.ReceivedM5[s.CurClientId-1] = false
	s.ClientsMsgStatus.ReceivedM5[s.CurClientId] = false
	s.ClientsMsgStatus.Unlock()

	// discard s.CurClientId - 1's data records and request a resend since the records might be incomplete
	s.Database.Lock()
	s.Database.Table[s.CurClientId-1] = s.Database.Table[s.CurClientId-1][:0]
	s.Database.Unlock()

	msg := Message{
		Id:      8,
		Desc:    "Send records",
		Content: map[string]any{"sendAll": true},
	}

	// send m8 (with sendAll=true) to s.CurClientId-1 as well
	sendMessage([]int{s.CurClientId - 1}, s.Server, msg, "message")

	msg = Message{
		Id:      8,
		Desc:    "Send records",
		Content: map[string]any{"sendAll": false},
	}
	sendMessage([]int{s.CurClientId}, s.Server, msg, "message")
	return
}

func (a *S4Action) Execute(s *ServerStateMachine) {
	// clear ServerStates flag
	s.ServerStates.MatchingDone = false
	s.ServerStates.MatchOccurs = false
	s.ServerStates.MatchOccursLastRecord = false
	// call matching function
	//fmt.Println("Perform matching...")
	if s.useLsh {
		matcherLSH(s)
	} else {
		matcher(s)
	}
}

func (a *S5Action) Execute(s *ServerStateMachine) {
	clearRecords(s)

	recordIds := getMatchingIds(s)
	msg := Message{
		Id:      3,
		Desc:    "Match",
		Content: map[string]any{"endFlag": false, "recordIds": recordIds},
	}
	clientList := rangeList(1, s.TotalNumClients)
	sendMessage(clientList, s.Server, msg, "message")
	return
}

func (a *S5aAction) Execute(s *ServerStateMachine) {
	s.CurClientId -= 1
	return
}

func (a *S6Action) Execute(s *ServerStateMachine) {
	clearRecords(s)

	recordIds := getMatchingIds(s)
	msg := Message{
		Id:      3,
		Desc:    "Match",
		Content: map[string]any{"endFlag": false, "recordIds": recordIds},
	}

	clientList := rangeList(1, s.TotalNumClients)
	sendMessage(clientList, s.Server, msg, "message")
	return
}

func (a *S6aAction) Execute(s *ServerStateMachine) {
	clearRecords(s)

	recordIds := getMatchingIds(s)
	msg := Message{
		Id:      3,
		Desc:    "Match",
		Content: map[string]any{"endFlag": true, "recordIds": recordIds},
	}

	clientList := rangeList(1, s.TotalNumClients)
	sendMessage(clientList, s.Server, msg, "message")
	return
}

func (a *S7Action) Execute(s *ServerStateMachine) {
	//fmt.Println(s.MatchingResults)
	log.Printf("Total # of performed matchings: %d\n", s.NumMatchCount)
	log.Printf("Total time elapsed: %s\n", time.Since(startTime))
	//fmt.Println(s.EventTriggered)
	// write to csv
	fmt.Println("Writing results to csv...")
	csvFile, err := os.Create(s.outputPath)

	if err != nil {
		log.Printf("Failed creating file: %s\n", err)
	}
	csvwriter := csv.NewWriter(csvFile)

	// sort the keys to make the csv line order consistent
	curIds := make([]int, 0)
	for k := range s.MatchingResults {
		curIds = append(curIds, k)
	}
	sort.Ints(curIds)

	_ = csvwriter.Write([]string{"ClientID", "RecordId", "ClientID", "RecordId", "Match"})
	for _, curId := range curIds {
		for prevId := range s.MatchingResults[curId][0] {
			for curRecord := range s.MatchingResults[curId] {
				for prevRecord := range s.MatchingResults[curId][curRecord][prevId] {
					var row []string
					// The record IDs we output start from 1
					if s.MatchingResults[curId][curRecord][prevId][prevRecord] {
						row = []string{strconv.Itoa(curId), strconv.Itoa(curRecord + 1),
							strconv.Itoa(prevId + 1), strconv.Itoa(prevRecord + 1), "1"}
					} else {
						row = []string{strconv.Itoa(curId), strconv.Itoa(curRecord + 1),
							strconv.Itoa(prevId + 1), strconv.Itoa(prevRecord + 1), "0"}
					}
					_ = csvwriter.Write(row)
				}
			}
		}
	}
	csvwriter.Flush()
	err = csvFile.Close()
	if err != nil {
		log.Printf("Failed closing file: %s\n", err)
	}

	msg := Message{
		Id:      4,
		Desc:    "End session",
		Content: nil,
	}

	clientList := rangeList(1, s.TotalNumClients)
	sendMessage(clientList, s.Server, msg, "message")
	fmt.Println("Done")
	return
}

// clear all the records in the database, clear the LSHTable, and increase CurKeyId by 1
func clearRecords(s *ServerStateMachine) {
	s.Database.Lock()
	defer s.Database.Unlock()
	// clear all data records
	for k := range s.Database.Table {
		s.Database.Table[k] = []*Record{}
	}
	for i := range s.Database.LSHTable {
		s.Database.LSHTable[i] = map[int][]RecordIndex{}
	}
	s.Database.CurKeyId += 1
}
