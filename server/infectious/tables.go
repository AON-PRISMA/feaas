package infectious

import (
	"fmt"
	"github.com/kshedden/gonpy"
	"github.com/sbinet/npyio"
	"github.com/schollz/progressbar/v3"
	"os"
	"path"
)

var (
	gf_exp = []uint32{}
	gf_log = []uint32{}
)

func check(e error) {
	if e != nil {
		panic(e)
	}
}

func mul(a, b MyInt) MyInt {
	if a == 0 || b == 0 {
		return MyInt(0)
	}
	log_a := int(gf_log[a])
	log_b := int(gf_log[b])
	return MyInt(gf_exp[(log_a+log_b)%fieldsize])
}

func inv(a MyInt) MyInt {
	if a == 0 {
		return MyInt(0)
	}
	return MyInt(gf_exp[uint32(fieldsize)-gf_log[a]])
}

//reads a numpy file containing 32 bit lookup values using gonpy reader for uint32, and returns it asynchronously as a vector slice

func asynchreadernew(filename string, ptr *[]uint32, err chan error) {
	r, erro := gonpy.NewFileReader(filename)
	*ptr, _ = r.GetUint32()
	err <- erro
}

type params struct {
	p uint64
	k uint32
	g uint32
}

func mul_for_table(x, y, poly, k uint32) uint32 {
	var hibit uint32 = 1 << (k - 1)
	var p uint32 = 0
	for i := uint(0); i < uint(k); i++ {
		if (y & 1) != 0 {
			p ^= x
		}
		wasset := (x & hibit) != 0
		x <<= 1
		y >>= 1
		if wasset {
			x ^= poly
		}
	}
	return p
}

//generates a set lookup tables for exp and log and stores them in numpy format

func GentableNPIO(n, p int, g uint32, k uint32) { // p should be the irreducible poly and g should be the generator. k should be set to the number of bits in the field
	m := n - 1
	if p < n || p >= 2*n {
		println("ErrPoly is out of range")
	}
	if g == 0 || g == 1 {
		println("Generator is not valid")
	}

	params := params{
		p: uint64(p),
		k: k,
		g: g,
	}

	log := make([]uint32, n)
	//exp := make([]uint32, 2*n-2)
	exp := make([]uint32, n)

	// Use the generator to compute the exp/log tables.  We perform the
	// usual trick of doubling the exp table to simplify Mul.
	var x uint32 = 1
	bar := progressbar.Default(int64(m))
	for i := 0; i < m; i++ {
		if x == 1 && i != 0 {
			println("this is not a generator")
		}
		exp[i] = x
		if i+m < n {
			exp[i+m] = x
		}
		log[x] = uint32(i)

		x = mul_for_table(x, g, uint32(p), params.k)
		bar.Add(1)
	}
	log[0] = uint32(m)
	dir := "tables"

	os.Mkdir(dir, 0777)

	fileName1 := path.Join(dir, fmt.Sprintf("log%d.npy", k))
	fileName2 := path.Join(dir, fmt.Sprintf("exp%d.npy", k))
	f1, err := os.Create(fileName1)
	check(err)
	f2, err := os.Create(fileName2)
	check(err)
	defer f1.Close()
	defer f2.Close()

	fmt.Println("Writing tables to disk...")
	err = npyio.Write(f1, log)
	check(err)
	err = npyio.Write(f2, exp)
	check(err)

}

func init() {
	fmt.Println("Initializing the server...")
	if bitlength == "16" {

		//check if the table(s) exists, and if not generate the tables.

		_, err := os.Stat("tables/exp16.npy")

		if os.IsNotExist(err) {
			fmt.Println("Generating 16-bit tables")
			GentableNPIO(65536, 0x1100b, 0x02, 16)
		}

		ch := make(chan error, 100)

		go asynchreadernew("tables/exp16.npy", &gf_exp, ch)
		go asynchreadernew("tables/log16.npy", &gf_log, ch)
		check(<-ch)
		check(<-ch)
	}

	if bitlength == "32" {
		//check if the table(s) exists, and if not generate the tables.
		_, err := os.Stat("tables/exp32.npy")

		if os.IsNotExist(err) {
			fmt.Println("Generating 32-bit tables")
			GentableNPIO(4294967296, 0x100400007, 0x02, 32)
		}

		ch := make(chan error, 100)

		go asynchreadernew("tables/exp32.npy", &gf_exp, ch)
		go asynchreadernew("tables/log32.npy", &gf_log, ch)
		check(<-ch)
		check(<-ch)
	}
	fmt.Println("Loading has completed.")
}
