@echo off

if exist "build\" (
	REM call "build/program.exe"
	pushd data 
	start ../build/program.exe
	popd
) else (
	echo /build/ does not exist
)

