package autograd

import (
	"fmt"
	"log"
	"path/filepath"
	"reflect"
	_ "reflect"
	"runtime"
	"strings"
)

func logInfo(formating string, args ...interface{}) {
	filename, line, funcname := "???", 0, "???"
	pc, filename, line, ok := runtime.Caller(1)
	if ok {
		funcname = runtime.FuncForPC(pc).Name()
		funcname = filepath.Ext(funcname)
		funcname = strings.TrimPrefix(funcname, ".")

		filename = filepath.Base(filename)
	}

	log.Printf("%s [%s] line %d: %s\n", filename, funcname, line, fmt.Sprintf(formating, args...))
}

func sum(arr ...float64) float64 {
	res := 0.0
	for _, x := range arr {
		res += x
	}

	return res
}

func getFuncName(f interface{}) string {
	return runtime.FuncForPC(reflect.ValueOf(f).Pointer()).Name()
}

func getVauleType(value interface{}) string {
	return fmt.Sprintf("%T", v)
}
