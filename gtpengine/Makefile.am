noinst_LIBRARIES = libfuego_gtpengine.a

libfuego_gtpengine_a_SOURCES = \
GtpEngine.cpp \
GtpInputStream.cpp \
GtpOutputStream.cpp

noinst_HEADERS = \
GtpEngine.h \
GtpInputStream.h \
GtpOutputStream.h

fuego_LDADD = \
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
$(BOOST_THREAD_LIB)

libfuego_gtpengine_a_CPPFLAGS = $(BOOST_CPPFLAGS)

DISTCLEANFILES = *~
