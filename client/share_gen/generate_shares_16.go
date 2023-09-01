// The MIT License (MIT)
//
// Copyright (C) 2016-2017 Vivint, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//go:build 16

package main

// #include <stdlib.h>
import "C"
import (
	"math/rand"
	"time"
	"unsafe"
)

type MyInt uint16 // changes the datatype used in the matching protocol. The bitlength of the type needs to match the bitlength variable set below.
type gfVal MyInt
type gfPoly []gfVal

// changes the bitlength of the matching protocol, choose between 8, 16 and 32.
// In addition to this, the user needs to include/exclude the appropriate lookup table gofile for the change to work.
var bitlength = "16"

func gfConst(val MyInt) gfVal {
	return gfVal(val)
}

func (a gfVal) mul(b gfVal) gfVal {
	if a == 0 || b == 0 {
		return gfVal(0)
	}
	log_i := int(gf_log[a])
	log_j := int(gf_log[b])
	return gfVal(gf_exp[(log_i + log_j)])
}

func (b gfVal) pow(val int) gfVal {
	out := gfVal(1)
	for i := 0; i < val; i++ {
		out = gfVal(b.mul(out))
	}
	return out
}

func polyZero(size int) gfPoly {
	out := make(gfPoly, size)
	for i := range out {
		out[i] = gfConst(0)
	}
	return out
}

func (p gfPoly) deg() int {
	return len(p) - 1
}

func (p gfPoly) index(power int) gfVal {
	if power < 0 {
		return gfConst(0)
	}
	which := p.deg() - power
	if which < 0 {
		return gfConst(0)
	}
	return p[which]
}

func (p *gfPoly) set(pow int, coef gfVal) {
	which := p.deg() - pow
	if which < 0 {
		*p = append(polyZero(-which), *p...)
		which = p.deg() - pow
	}
	(*p)[which] = coef
}

func (a gfVal) add(b gfVal) gfVal {
	return gfVal(a ^ b)
}

func (p gfPoly) add(b gfPoly) gfPoly {
	size := len(p)
	if lb := len(b); lb > size {
		size = lb
	}
	out := make(gfPoly, size)
	for i := range out {
		pi := p.index(i)
		bi := b.index(i)
		out.set(i, pi.add(bi))
	}
	return out
}

func (p gfPoly) eval(x gfVal) gfVal {
	out := gfConst(0)
	for i := 0; i <= p.deg(); i++ {
		x_i := x.pow(i)
		p_i := p.index(i)
		out = out.add(p_i.mul(x_i))
	}
	return out
}

//export generateShares
func generateShares(total, required int) *C.ushort { // generates a random polynomial, and then generates shares from that polynomial
	polynomial := make(gfPoly, required)
	const interp_base = gfVal(2)
	rand.Seed(time.Now().UnixNano()) // to ensure polynomial has different coefficients each run
	for i := range polynomial {
		polynomial[i] = gfConst(MyInt(rand.Uint64()))
	}
	//var ret []MyInt
	p := C.malloc(C.size_t(total) * C.size_t(unsafe.Sizeof(C.ushort(0))))
	ret := (*[1<<30 - 1]C.ushort)(p)[:total:total]
	for i := 0; i < total; i++ {
		pt := gfConst(0)
		if i != 0 {
			pt = interp_base.pow(i - 1)
		}
		//ret = append(ret, MyInt(polynomial.eval(pt)))
		ret[i] = C.ushort(MyInt(polynomial.eval(pt)))
	}
	return (*C.ushort)(p) // C.ushort is equivalent to uint16
}

//export freePtr
func freePtr(ptr *C.ushort) {
	C.free(unsafe.Pointer(ptr))
}

func main() {}
