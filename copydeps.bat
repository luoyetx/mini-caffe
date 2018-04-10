call 3rdparty\src\protobuf\cmake\build\extract_includes.bat
move include\google 3rdparty\include\
copy 3rdparty\src\protobuf\cmake\build\Debug\libprotobuf-lited.lib 3rdparty\lib\libprotobuf-lited.lib
copy 3rdparty\src\protobuf\cmake\build\Release\libprotobuf-lite.lib 3rdparty\lib\libprotobuf-lite.lib
copy 3rdparty\src\protobuf\cmake\build\Release\protoc.exe 3rdparty\bin\protoc.exe