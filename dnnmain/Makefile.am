bin_PROGRAMS = dnnmain

dnnmain_SOURCES = DnnMain.cpp \
DnnEngine.cpp

noinst_HEADERS = \
DnnEngine.h

dnnmain_LDFLAGS = $(BOOST_LDFLAGS)

dnnmain_LDADD = \
../gouct/libfuego_gouct.a \
../simpleplayers/libfuego_simpleplayers.a \
../go/libfuego_go.a \
../features/libfuego_features.a \
../smartgame/libfuego_smartgame.a \
../gtpengine/libfuego_gtpengine.a \
$(BOOST_PROGRAM_OPTIONS_LIB) \
$(BOOST_SYSTEM_LIB) \
$(BOOST_FILESYSTEM_LIB) \
$(BOOST_THREAD_LIB)

dnnmain_DEPENDENCIES = \
../gouct/libfuego_gouct.a \
../simpleplayers/libfuego_simpleplayers.a \
../go/libfuego_go.a \
../smartgame/libfuego_smartgame.a \
../gtpengine/libfuego_gtpengine.a \
../features/libfuego_features.a

dnnmain_CPPFLAGS = \
$(BOOST_CPPFLAGS) \
-I@top_srcdir@/gtpengine \
-I@top_srcdir@/smartgame \
-I@top_srcdir@/go \
-I@top_srcdir@/simpleplayers \
-I@top_srcdir@/gouct \
-I@top_srcdir@/features 

DISTCLEANFILES = *~
