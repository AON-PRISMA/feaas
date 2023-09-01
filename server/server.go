package main

import (
	"Server/state_machine"
	"Server/utils"
	"flag"
	"fmt"
	socketio "github.com/googollee/go-socket.io"
	"github.com/googollee/go-socket.io/engineio"
	"io"
	"log"
	"net/http"
	"sort"
	"strconv"
	"time"
)

func main() {
	// command line args:
	totalNumClients := flag.Int("num_c", 2, "total number of participating clients")
	rep := flag.Int("rep", 1, "the repetition param for the encryption")
	lshRep := flag.Int("lsh_rep", 10, "number of repetitions for encryption to improve security")

	plain := flag.Int("plain", 0, "clients' records are plain texts or not. For testing purposes")

	tls := flag.Int("tls", 1, "whether to enable tls; 1 or 0")

	addr := flag.String("addr", "127.0.0.1:8000", "ip address the server is listening")

	output := flag.String("output", "result.csv", "the output csv path for storing matching results")

	debug := flag.Int("debug", 0, "whether to enable debug mode")
	flag.Parse()

	if *debug == 0 {
		log.SetOutput(io.Discard)
	}

	var socketConfig = &engineio.Options{

		PingTimeout: 60 * 20 * time.Second,
	}

	server := socketio.NewServer(socketConfig)
	stateMachine := state_machine.NewServerStateMachine(server, *totalNumClients, *rep, *lshRep, *plain, *output)
	go stateMachine.Run()

	server.OnConnect("/", func(s socketio.Conn) error {
		s.SetContext("")
		s.Join(s.ID()) // Assign a room for each connected clients.
		fmt.Printf("Client %s connected\n", s.ID())
		return nil
	})
	server.OnEvent("/", "data_record", func(s socketio.Conn, val any) {
		cId, err := strconv.Atoi(s.ID())
		if err != nil {
			panic(err)
		}
		records := map[int]state_machine.Record{}
		utils.ParseMessage(val, &records)
		stateMachine.Database.Lock()
		defer stateMachine.Database.Unlock()
		// loop records in order
		keys := make([]int, 0, len(records))
		for k := range records {
			keys = append(keys, k)
		}
		sort.Ints(keys)
		for i, k := range keys {
			//stateMachine.Database.Table[cId] = utils.SetIdx(stateMachine.Database.Table[cId], k, &v)
			v := records[k]
			// Discard records with a non-correct key id
			if v.KeyId < stateMachine.Database.CurKeyId {
				continue
			}
			v.RecordId = k
			stateMachine.Database.Table[cId] = append(stateMachine.Database.Table[cId], &v)
			if v.LastRecord {
				stateMachine.ClientsMsgStatus.Lock()
				stateMachine.ClientsMsgStatus.ReceivedM5[cId] = true
				stateMachine.ClientsMsgStatus.Unlock()
			}
			// update LSHTable for cur record
			for j := 0; j < len(stateMachine.Database.LSHTable); j++ {
				if recordList, found := stateMachine.Database.LSHTable[j][v.LSHVal[j]]; found {
					stateMachine.Database.LSHTable[j][v.LSHVal[j]] = append(recordList,
						state_machine.RecordIndex{ClientIndex: cId, RecordIndex: i})
				} else {
					stateMachine.Database.LSHTable[j][v.LSHVal[j]] = []state_machine.RecordIndex{{ClientIndex: cId, RecordIndex: i}}
				}

			}
		}
		//fmt.Println(stateMachine.Database.Table)
		//fmt.Println(len(stateMachine.Database.Table))
		//s.Emit("reply", records)
	})

	server.OnEvent("/", "message", func(s socketio.Conn, val any) {
		cId, err := strconv.Atoi(s.ID())
		if err != nil {
			panic(err)
		}
		msg := state_machine.Message{}
		utils.ParseMessage(val, &msg)
		stateMachine.ClientsMsgStatus.Lock()
		defer stateMachine.ClientsMsgStatus.Unlock()
		if msg.Id == 1 {
			if len(msg.Content) != 0 {
				stateMachine.Threshold = int(msg.Content["th"].(float64))
				fmt.Printf("Server using threshold: %d\n", stateMachine.Threshold)
			}
			stateMachine.ClientsMsgStatus.ReceivedM1[cId] = true
		} else if msg.Id == 7 {
			stateMachine.ClientsMsgStatus.ReceivedM7[cId] = true
		} else {
			// error handling
			log.Fatal("Wrong message id.")
		}
	})

	server.OnError("/", func(s socketio.Conn, e error) {
		log.Fatal("meet error:", e)
	})

	server.OnDisconnect("/", func(s socketio.Conn, reason string) {
		fmt.Printf("Client %s disconnected\n", s.ID())
	})

	go func() {
		err := server.Serve()
		if err != nil {
			log.Fatal(err)
		}
	}()
	defer func() {
		err := server.Close()
		if err != nil {
			log.Fatal(err)
		}
	}()

	http.Handle("/socket.io/", server)
	log.Printf("Serving at %s...", *addr)
	if *tls != 0 {
		log.Fatal(http.ListenAndServeTLS(*addr, "localhost+1.pem", "localhost+1-key.pem", nil))
	} else {
		log.Fatal(http.ListenAndServe(*addr, nil))
	}
}
