
ROOTCFLAGS   := $(shell root-config --cflags)
ROOTLIBS     := $(shell root-config --libs)

CXX           = g++
CXXFLAGS      = -Wall -fPIC -O3 -march=native
LD            = g++
LDFLAGS       = -O3 -march=native

CXXFLAGS     += $(ROOTCFLAGS)
LIBS          = $(ROOTLIBS) $(SYSLIBS)

vpath %.cpp src
objdir     = obj

SRC        = cll.cpp eos.cpp eoChiral_s.cpp eoHadron.cpp trancoeff.cpp fld.cpp hdo.cpp main.cpp rmn.cpp cornelius.cpp \
             icTrento.cpp eos_hotqcd.cpp fast.cpp
OBJS       = $(patsubst %.cpp,$(objdir)/%.o,$(SRC)) 
              
TARGET	   = hlle_visc
#------------------------------------------------------------------------------
$(TARGET):       $(OBJS)
		$(LD)  $(LDFLAGS) $^ -o $@ $(LIBS)
		@echo "$@ done"
clean:
		@rm -f $(OBJS) $(TARGET)

$(OBJS): | $(objdir)

$(objdir):
	@mkdir -p $(objdir)
	
obj/%.o : %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
