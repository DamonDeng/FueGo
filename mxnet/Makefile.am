#AUTOMAKE_OPTIONS=foreign  

noinst_LIBRARIES = lib_mxnetmodel.a

lib_mxnetmodel_a_SOURCES = \
MXNetModel.cpp 

noinst_HEADERS = \
MXNetModel.h 

lib_mxnetmodel_a_CPPFLAGS = \
$(BOOST_CPPFLAGS) \
-I@top_srcdir@/smartgame \ 
-I@top_srcdir@/include \
-I@top_srcdir@/go \
-I@top_srcdir@/gouct \
-I@top_srcdir@/gtpengine \
-I@top_srcdir@/include



bin_PROGRAMS = \
MXNetExtract \
MXNetMain \
GoBoardTest

MXNetMain_SOURCES = MXNetMain.cpp

MXNetMain_LDFLAGS = $(BOOST_LDFLAGS)

MXNetMain_LDADD = \
../smartgame/libfuego_smartgame.a \
../gouct/libfuego_gouct.a \
../go/libfuego_go.a \
../lib/libmxnet.so \
../simpleplayers/libfuego_simpleplayers.a \
$(BOOST_PROGRAM_OPTIONS_LIB) \
$(BOOST_SYSTEM_LIB) \
$(BOOST_FILESYSTEM_LIB) \
$(BOOST_THREAD_LIB)

MXNetMain_DEPENDENCIES = \
../gouct/libfuego_gouct.a \
../go/libfuego_go.a \
../smartgame/libfuego_smartgame.a \
../simpleplayers/libfuego_simpleplayers.a

MXNetMain_CPPFLAGS = \
$(BOOST_CPPFLAGS) \
-I@top_srcdir@/smartgame \
-I@top_srcdir@/include \
-I@top_srcdir@/go \
-I@top_srcdir@/gouct \
-I@top_srcdir@/simpleplayers \
-I@top_srcdir@/include

MXNetExtract_SOURCES = feature_extract.cpp

MXNetExtract_LDFLAGS = $(BOOST_LDFLAGS)

MXNetExtract_LDADD = \
../smartgame/libfuego_smartgame.a \
../gouct/libfuego_gouct.a \
../go/libfuego_go.a \
../lib/libmxnet.so \
$(BOOST_PROGRAM_OPTIONS_LIB) \
$(BOOST_SYSTEM_LIB) \
$(BOOST_FILESYSTEM_LIB) \
$(BOOST_THREAD_LIB)

MXNetExtract_DEPENDENCIES = \
../gouct/libfuego_gouct.a \
../go/libfuego_go.a \
../smartgame/libfuego_smartgame.a 

MXNetExtract_CPPFLAGS = \
$(BOOST_CPPFLAGS) \
-I@top_srcdir@/smartgame \
-I@top_srcdir@/include \
-I@top_srcdir@/go \
-I@top_srcdir@/gouct 



GoBoardTest_SOURCES = GoBoardTest.cpp

GoBoardTest_LDFLAGS = $(BOOST_LDFLAGS)

GoBoardTest_LDADD = \
../smartgame/libfuego_smartgame.a \
../gouct/libfuego_gouct.a \
../go/libfuego_go.a \
../lib/libmxnet.so \
$(BOOST_PROGRAM_OPTIONS_LIB) \
$(BOOST_SYSTEM_LIB) \
$(BOOST_FILESYSTEM_LIB) \
$(BOOST_THREAD_LIB)

GoBoardTest_DEPENDENCIES = \
../gouct/libfuego_gouct.a \
../go/libfuego_go.a \
../smartgame/libfuego_smartgame.a 

GoBoardTest_CPPFLAGS = \
$(BOOST_CPPFLAGS) \
-I@top_srcdir@/smartgame \
-I@top_srcdir@/include \
-I@top_srcdir@/go \
-I@top_srcdir@/gouct \
-I@top_srcdir@/include





DISTCLEANFILES = *~

