bin_PROGRAMS = selfplay

selfplay_SOURCES = \
SelfplayMain.cpp 

noinst_HEADERS = \
SelfplayMain.h

selfplay_LDFLAGS = $(BOOST_LDFLAGS)

selfplay_LDADD = \
../gouct/libfuego_gouct.a \
../go/libfuego_go.a \
../features/libfuego_features.a \
../smartgame/libfuego_smartgame.a \
../gtpengine/libfuego_gtpengine.a \
../lib/libmxnet.so \
../mxnet/lib_mxnetmodel.a \
$(BOOST_PROGRAM_OPTIONS_LIB) \
$(BOOST_FILESYSTEM_LIB) \
$(BOOST_SYSTEM_LIB) \
$(BOOST_THREAD_LIB) \
 -lpthread

selfplay_DEPENDENCIES = \
../gouct/libfuego_gouct.a \
../go/libfuego_go.a \
../features/libfuego_features.a \
../smartgame/libfuego_smartgame.a \
../gtpengine/libfuego_gtpengine.a \
../lib/libmxnet.so \
../mxnet/lib_mxnetmodel.a

selfplay_CPPFLAGS = \
-DDATADIR='"@datadir@"' \
-DABS_TOP_SRCDIR='"@abs_top_srcdir@"' \
$(BOOST_CPPFLAGS) \
-I@top_srcdir@/gtpengine \
-I@top_srcdir@/smartgame \
-I@top_srcdir@/features \
-I@top_srcdir@/go \
-I@top_srcdir@/gouct \
-I@top_srcdir@/mxnet \
-I@top_srcdir@/include

DISTCLEANFILES = *~
