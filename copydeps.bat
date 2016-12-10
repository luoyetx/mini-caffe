mkdir 3rdparty\include\glog
copy 3rdparty\src\glog\src\windows\glog 3rdparty\include\glog
copy 3rdparty\src\glog\Debug\libglog.lib 3rdparty\lib\libglogd.lib
copy 3rdparty\src\glog\Release\libglog.lib 3rdparty\lib\libglog.lib
copy 3rdparty\src\glog\Release\libglog.dll 3rdparty\bin\libglog.dll

call 3rdparty\src\protobuf\cmake\build\extract_includes.bat
move include\google 3rdparty\include\
copy 3rdparty\src\protobuf\cmake\build\Debug\libprotobuf.lib 3rdparty\lib\libprotobufd.lib
copy 3rdparty\src\protobuf\cmake\build\Release\libprotobuf.lib 3rdparty\lib\libprotobuf.lib
copy 3rdparty\src\protobuf\cmake\build\Release\protoc.exe 3rdparty\bin\protoc.exe