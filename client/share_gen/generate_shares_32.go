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

//go:build !16

package main

// #include <stdlib.h>
import "C"
import (
	"math/rand"
	"time"
	"unsafe"
)

type gfVal MyInt

type MyInt uint32 // changes the datatype used in the matching protocol. The bitlength of the type needs to match the bitlength variable set below.

type gfPoly []gfVal

var bitlength = "32" // changes the bitlength of the matching protocol, choose between 8, 16 and 32.

func gfConst(val MyInt) gfVal {
	return gfVal(val)
}

func (b gfVal) pow(val int) gfVal {
	out := gfVal(1)
	for i := 0; i < val; i++ {
		out = gfVal(b.mul(out))
	}
	return out
}

func (a gfVal) mul(b gfVal) gfVal {
	const irreducable = uint64(0x1000400007) // we use 0x011D which is x^8 + x^4 + x^3 + x^2 + 1 for 8 bits, we use 0x100000000000001B   which is x^64 + x^4 + x^3 + x + 1 for 64 bits, and 0x1000400007   x^32 + x^22 + x^2 + x^1 + 1 for 32 bits

	//fmt.Printf("%b", irreducable)
	var result uint64
	if a == 0 || b == 0 {
		return gfVal(gfConst(0))
	}
	int_a := uint64(a)
	int_b := uint64(b)
	result = 0
	for i := 0; i < 32; i++ {
		result = result << 1
		if (result & 0x100000000) != 0 {
			result = result ^ irreducable
		}
		if (int_b & 0x080000000) != 0 {
			result = result ^ int_a
		}
		int_b = int_b << 1
	}
	return gfVal(result)
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
func generateShares(total, required int) *C.uint { // generates a random polynomial, and then generates shares from that polynomial
	polynomial := make(gfPoly, required)
	const interp_base = gfVal(2)
	rand.Seed(time.Now().UnixNano()) // to ensure polynomial has different coefficients each run
	for i := range polynomial {
		polynomial[i] = gfConst(MyInt(rand.Uint64()))
	}
	//var ret []MyInt
	p := C.malloc(C.size_t(total) * C.size_t(unsafe.Sizeof(C.uint(0))))
	ret := (*[1<<30 - 1]C.uint)(p)[:total:total]
	for i := 0; i < total; i++ {
		pt := gfConst(0)
		if i != 0 {
			pt = interp_base.pow(i - 1)
		}
		//ret = append(ret, MyInt(polynomial.eval(pt)))
		ret[i] = C.uint(MyInt(polynomial.eval(pt)))
	}
	return (*C.uint)(p)
}

//export freePtr
func freePtr(ptr *C.uint) {
	C.free(unsafe.Pointer(ptr))
}

func main() {}
