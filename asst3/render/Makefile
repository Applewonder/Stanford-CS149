EXECUTABLE := render

CU_FILES   := cudaRenderer.cu benchmark.cu
CC_FILES   := main.cpp display.cpp refRenderer.cpp \
              noise.cpp ppm.cpp sceneLoader.cpp

OBJDIR=objs
CXXFLAGS=-O3 -Wall -g3
# -----------------------------------------------
# 启用 -rdc=true 并指定 arch>=3.5(你这里写的是 sm_89)
NVCCFLAGS=-O3 -m64 -g -G -arch=sm_89 -rdc=true
# -----------------------------------------------
# 要链接的库
LIBS += GL glut cudart cudadevrt

ifneq ($(wildcard /opt/cuda-8.0/.*),)
    LDFLAGS=-L/opt/cuda-8.0/lib64/ -lcudart
else
    LDFLAGS=-L/usr/local/cuda/lib64/ -lcudart
endif

LDLIBS  := $(addprefix -l, $(LIBS))

NVCC=nvcc

OBJS = $(OBJDIR)/main.o $(OBJDIR)/display.o $(OBJDIR)/benchmark.o \
       $(OBJDIR)/refRenderer.o $(OBJDIR)/cudaRenderer.o $(OBJDIR)/noise.o \
       $(OBJDIR)/ppm.o $(OBJDIR)/sceneLoader.o

.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
	@mkdir -p $(OBJDIR)/

clean:
	rm -rf $(OBJDIR) *~ $(EXECUTABLE) logs

check: default
	./checker.pl

# ------------------------------
# 使用 nvcc 而不是 g++ 来做最终链接
# 并带上 -rdc=true、-arch=sm_XX 等
# ------------------------------
$(EXECUTABLE): dirs $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS)

# ------------------------------
# 编译 .cpp -> .o
# ------------------------------
$(OBJDIR)/%.o: %.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

# ------------------------------
# 编译 .cu -> .o
# (已在 NVCCFLAGS 含 -rdc=true)
# ------------------------------
$(OBJDIR)/%.o: %.cu
	$(NVCC) $< $(NVCCFLAGS) -c -o $@
