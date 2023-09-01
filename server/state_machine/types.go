package state_machine

import (
	"Server/infectious"
	"sync"
)

// Database storage

type Record struct {
	DataRecord []infectious.MyInt `json:"data_record"` // the encoding vector
	LSHVal     []int              `json:"lsh_val"`     // the lsh values
	RecordId   int                `json:"record_id"`   // the record_id
	KeyId      int                `json:"key_id"`      // the encryption key id
	LastRecord bool               `json:"last_record"` // whether the record is the last record
}

type Database_ struct {
	Table    map[int][]*Record // a mapping from client id to list of *Record
	LSHTable LSHTable_         // a data structure containing record info for different lsh bins
	CurKeyId int               // current key id, used for rejecting clients' records
	sync.RWMutex
}

type ClientsMsgStatus_ struct {
	ReceivedM1 map[int]bool // a mapping from client id to bool
	ReceivedM5 map[int]bool
	ReceivedM7 map[int]bool
	sync.Mutex
}

type ServerStates_ struct {
	MatchOccurs           bool
	MatchOccursLastRecord bool
	MatchingDone          bool
}

type Message struct {
	Id      int            `json:"id"`
	Desc    string         `json:"desc"`
	Content map[string]any `json:"content"`
}

type MatchingInfo struct {
	StreamingClientId  int `json:"streaming_client_id"`
	StreamingClientInd int `json:"streaming_client_ind"`
	MatchingClientId   int `json:"matching_client_id"`
	MatchingClientInd  int `json:"matching_client_ind"`
}

type RecordIndex struct {
	ClientIndex int // client id
	RecordIndex int // record id
}

type LSHTable_ []map[int][]RecordIndex
