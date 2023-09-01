go get
go build -tags $1 -o server server.go
cd ../client/share_gen
go build -tags $1 -buildmode=c-shared -o _share_gen.so
