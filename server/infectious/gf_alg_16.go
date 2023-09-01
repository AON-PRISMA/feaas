//go:build 16

package infectious

import (
	"errors"
)

//
//Contains parameters and helpers specific to gf(2^16) operations
//

type gfVal MyInt

type MyInt uint16 // changes the datatype used in the matching protocol. The bitlength of the type needs to match the bitlength variable set below.

var bitlength = "16" // changes the bitlength of the matching protocol, choose between 16 and 32.

var fieldsize = 65535 // fieldsize parameter used for inverse

func (a gfVal) inv() (gfVal, error) {
	if a == 0 {
		return 0, errors.New("invert zero")
	}
	return gfVal(gf_exp[65535-gf_log[a]]), nil

}
