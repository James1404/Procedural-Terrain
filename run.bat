@echo off

if exist "build\" (
	REM call "build/program.exe"
	pushd data 
	call "../build/program.exe"
	::start ../build/program.exe
	popd
) else (
	echo /build/ does not exist
)

