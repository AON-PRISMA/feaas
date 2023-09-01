package state_machine

// State and event Ids
const (
	s0Id  StateId = 0
	s1Id  StateId = 1
	s2Id  StateId = 2
	s3Id  StateId = 3
	s3aId StateId = 4
	s4Id  StateId = 5
	s5Id  StateId = 6
	s5aId StateId = 7
	s6Id  StateId = 8
	s6aId StateId = 9
	s7Id  StateId = 10

	e0Id  EventId = 0
	e1Id  EventId = 1
	e2Id  EventId = 2
	e3Id  EventId = 3
	e4Id  EventId = 4
	e5Id  EventId = 5
	e6Id  EventId = 6
	e7Id  EventId = 7
	e8Id  EventId = 8
	e9Id  EventId = 9
	e10Id EventId = 10
	e11Id EventId = 11
	e12Id EventId = 12
	e13Id EventId = 13
	e14Id EventId = 14
)

// defn of events
var e0 = Event{Dest: s1Id, Desc: "m1 received from all clients"}
var e1 = Event{Dest: s2Id, Desc: "null"}
var e2 = Event{Dest: s3Id, Desc: "received m7 from C1...Cn"}
var e3 = Event{Dest: s4Id, Desc: "C1 to C(i-1)'s m5 received"}
var e4 = Event{Dest: s5Id, Desc: "Match occurs, C1 to C(i-1) have all been iterated"}
var e5 = Event{Dest: s6Id, Desc: "Match occurs on the last record, C1 to C(i-1) have all been iterated, i<n"}
var e6 = Event{Dest: s3aId, Desc: "matching is done, m5 from Ci is received. i < n"}
var e7 = Event{Dest: s7Id, Desc: "matching is done, m5 from Ci is received. i == n"}
var e8 = Event{Dest: s5aId, Desc: "null"}
var e9 = Event{Dest: s4Id, Desc: "Ci and C(i-1)s' m5 received"}
var e10 = Event{Dest: s3Id, Desc: "null"}
var e11 = Event{Dest: s3Id, Desc: "null"}
var e12 = Event{Dest: s7Id, Desc: "null"}
var e13 = Event{Dest: s6aId, Desc: "Match occurs on the last record, C1 to C(i-1) have all been iterated, i==n"}
var e14 = Event{Dest: s0Id, Desc: "null"}

// defn of states
var s0 = State{Action: &S0Action{}, Events: Events{e0Id: e0}}
var s1 = State{Action: &S1Action{}, Events: Events{e1Id: e1}}
var s2 = State{Action: &S2Action{}, Events: Events{e2Id: e2}}
var s3 = State{Action: &S3Action{}, Events: Events{e3Id: e3}}
var s3a = State{Action: &S3aAction{}, Events: Events{e9Id: e9}}
var s4 = State{Action: &S4Action{}, Events: Events{e4Id: e4, e5Id: e5, e6Id: e6, e7Id: e7, e13Id: e13}}
var s5 = State{Action: &S5Action{}, Events: Events{e8Id: e8}}
var s5a = State{Action: &S5aAction{}, Events: Events{e10Id: e10}}
var s6 = State{Action: &S6Action{}, Events: Events{e11Id: e11}}
var s6a = State{Action: &S6aAction{}, Events: Events{e12Id: e12}}
var s7 = State{Action: &S7Action{}, Events: Events{e14Id: e14}}

var stateMap = States{
	s0Id:  s0,
	s1Id:  s1,
	s2Id:  s2,
	s3Id:  s3,
	s3aId: s3a,
	s4Id:  s4,
	s5Id:  s5,
	s5aId: s5a,
	s6Id:  s6,
	s6aId: s6a,
	s7Id:  s7,
}
