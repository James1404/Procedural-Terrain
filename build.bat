@echo off
if not defined DevEnvDir (
	call "E:\Microsoft Visual Studio\2019\VC\Auxiliary\Build\vcvarsall.bat" x64
)

set compiler_flags=/std:c++20 /W3 /Zi /FC /nologo /Od /Oi /WX /EHsc /Fo.\obj\ /Fd.\obj\ /Feprogram
set source_files=..\src\win32_main.cpp
set lib_files="..\includes\assimp\lib\assimp-vc142-mtd.lib"
set includes_folders=/I "..\includes\glm" /I "..\includes\assimp\include" /I "..\includes\stb" /I "../includes/fastnoiselite"
set defines=/DDEBUG=1

if not exist build\ (
	mkdir build
)

pushd build

if not exist obj\ (
	mkdir obj 
)

cl %compiler_flags% %defines% %includes_folders% %source_files% %lib_files%
popd
