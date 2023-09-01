package state_machine

import (
	"errors"
	socketio "github.com/googollee/go-socket.io"
	"log"
	"sync"
	"time"
)

// StateId represents an extensible state type in the state machine.
type StateId int

// EventId represents an extensible event type in the state machine.
type EventId int

// ErrEventRejected is the error returned when the state machine cannot process
// an event in the state that it is in.
var ErrEventRejected = errors.New("event rejected")

const (
	// Default represents the default state of the system.
	Default StateId = 0
)

// EventContext represents the context to be passed to the action implementation.
type EventContext struct {
	server *socketio.Server
}

// Action represents the action to be executed in a given state.
type Action interface {
	Execute(s *ServerStateMachine)
}

// Events represents a mapping of events and states.
type Events map[EventId]Event

// State binds a state with an action and a set of events it can handle.
type State struct {
	Action Action
	Events Events
}

type Event struct {
	Dest StateId
	Desc string
}

// States represents a mapping of states and their implementations.
type States map[StateId]State

// ServerStateMachine represents the state machine.
type ServerStateMachine struct {
	// Previous represents the previous state.
	Previous StateId

	// Current represents the current state.
	Current StateId

	// States holds the configuration of states and events handled by the state machine.
	StateMap States

	// mutex ensures that only 1 event is processed by the state machine at any given time.
	mutex sync.Mutex

	CurClientId int

	Server           *socketio.Server
	ClientsMsgStatus *ClientsMsgStatus_
	Database         *Database_
	ServerStates     *ServerStates_
	TotalNumClients  int

	MatchingResults map[int][][][]bool // data structure for storing the matching results

	NumMatchCount int // a counter that records number of total matching performed

	rep       int
	Threshold int
	useLsh    bool
	plain     int // for testing purposes

	outputPath string // the result output path

	//EventTriggered []bool // for testing purposes
}

func NewServerStateMachine(server *socketio.Server, totalNumClients int, rep int, lshRep int,
	plain int, outputPath string) *ServerStateMachine {
	s := &ServerStateMachine{Server: server, TotalNumClients: totalNumClients, NumMatchCount: 0, rep: rep,
		plain: plain, outputPath: outputPath}
	s.ClientsMsgStatus = &ClientsMsgStatus_{ReceivedM1: map[int]bool{}, ReceivedM5: map[int]bool{}, ReceivedM7: map[int]bool{}}
	s.ServerStates = &ServerStates_{}
	table := map[int][]*Record{}
	if lshRep < 0 {
		s.useLsh = false
		lshRep = 0
	} else {
		s.useLsh = true
	}
	lshTable := make(LSHTable_, lshRep)
	for i := range lshTable {
		lshTable[i] = map[int][]RecordIndex{}
	}
	s.Database = &Database_{Table: table, LSHTable: lshTable, CurKeyId: 0}
	// init the state machine here
	s.Previous = s0Id
	s.Current = s0Id
	s.StateMap = stateMap
	s.MatchingResults = map[int][][][]bool{}
	//s.EventTriggered = make([]bool, 15)
	return s
}

// check if a transition condition evaluates true
func (s *ServerStateMachine) checkCondition(eventId EventId) bool {
	s.ClientsMsgStatus.Lock()
	defer s.ClientsMsgStatus.Unlock()
	switch eventId {
	case e0Id:
		for i := 1; i <= s.TotalNumClients; i++ {
			if val, ok := s.ClientsMsgStatus.ReceivedM1[i]; !(ok && val) {
				return false
			}
		}
		return true
	case e2Id:
		for i := 1; i <= s.TotalNumClients; i++ {
			if val, ok := s.ClientsMsgStatus.ReceivedM7[i]; !(ok && val) {
				return false
			}
		}
		return true
	case e3Id:
		for i := 1; i <= s.CurClientId-1; i++ {
			if val, ok := s.ClientsMsgStatus.ReceivedM5[i]; !(ok && val) {
				return false
			}
		}
		return true
	case e4Id:
		return s.ServerStates.MatchingDone && s.ServerStates.MatchOccurs && !s.ServerStates.MatchOccursLastRecord
	case e5Id:
		return s.ServerStates.MatchingDone && s.ServerStates.MatchOccurs && s.ServerStates.MatchOccursLastRecord &&
			s.CurClientId < s.TotalNumClients
	case e13Id:
		return s.ServerStates.MatchingDone && s.ServerStates.MatchOccurs && s.ServerStates.MatchOccursLastRecord &&
			s.CurClientId == s.TotalNumClients
	case e6Id:
		return s.ServerStates.MatchingDone && !s.ServerStates.MatchOccurs && s.CurClientId < s.TotalNumClients
	case e7Id:
		return s.ServerStates.MatchingDone && !s.ServerStates.MatchOccurs && s.CurClientId == s.TotalNumClients
	case e9Id:
		return s.ClientsMsgStatus.ReceivedM5[s.CurClientId-1] && s.ClientsMsgStatus.ReceivedM5[s.CurClientId]
	case e1Id, e8Id, e10Id, e11Id, e12Id, e14Id:
		return true
	default:
		return false
	}
}

// getNextState returns the next state for the event given the machine's current
// state, or an error if the event can't be handled in the given state.
func (s *ServerStateMachine) getNextState() (StateId, error) {
	if state, ok := s.StateMap[s.Current]; ok {
		for {
			for eventId, event := range state.Events {
				if s.checkCondition(eventId) {
					//s.EventTriggered[eventId] = true
					return event.Dest, nil
				}
			}
			time.Sleep(100 * time.Millisecond)
		}
	}
	return Default, ErrEventRejected
}

// Run sends an event to the state machine.
func (s *ServerStateMachine) Run() {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	for {
		// Determine the next state for the event given the machine's current state.
		nextState, err := s.getNextState()
		if err != nil {
			panic(err)
		}

		// Identify the state definition for the next state.
		state, ok := s.StateMap[nextState]
		if !ok || state.Action == nil {
			panic("configuration error")
		}

		// Transition over to the next state.
		s.Previous = s.Current
		s.Current = nextState
		// Execute the next state's action
		log.Printf("server: Cur state ID: %d", s.Current)
		state.Action.Execute(s)
	}
}
