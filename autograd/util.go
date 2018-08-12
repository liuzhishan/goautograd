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

func getFuncKey(f interface{}) string {
	//return fmt.Sprintf("%#v", f)
	return getFuncName(f)
}

func getValueType(value interface{}) string {
	return fmt.Sprintf("%T", value)
}

func getNodeValue(boxed []Box) []interface{} {
	values := make([]interface{}, 0)
	for _, b := range boxed {
		values = append(values, b.value)
	}

	return values
}

func subvals(args []interface{}, argnum []int, values []interface{}) []interface{} {
	for i, x := range values {
		args[argnum[i]] = x
	}

	return args
}

func toFloat64(args []interface{}) []float64 {
	res := make([]float64, 0)
	for _, x := range args {
		res = append(res, x.(float64))
	}

	return res
}
