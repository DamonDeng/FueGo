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



DISTCLEANFILES = *~

