package utils

import (
	"encoding/json"
	"fmt"
)

func ParseMessage(msg any, ret any) {
	// TODO: need proper error handling
	jsonBody, err := json.Marshal(msg)
	if err != nil {
		// do error check
		fmt.Println(err)
	}
	if err := json.Unmarshal(jsonBody, ret); err != nil {
		// do error check
		fmt.Println(err)
	}
}
