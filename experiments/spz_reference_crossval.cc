// SPZ v4 cross-validation harness — preserved from the 2026-07-03 validation run.
//
// Validates src/aura/spz.py against the ACTUAL nianticlabs/spz reference core
// (github.com/nianticlabs/spz @ bb0efad, "Add in-memory I/O for python bindings").
// Build (no cmake needed; needs zstd headers + lib):
//   git clone https://github.com/nianticlabs/spz /tmp/spz-ref && cd /tmp/spz-ref && git checkout bb0efad
//   g++ -O2 -std=c++17 -I src/cc -I <zstd-include> experiments/spz_reference_crossval.cc \
//       src/cc/load-spz.cc src/cc/splat-types.cc src/cc/splat-c-types.cc -lzstd -o xval_harness
// Validation performed (all passed, see src/aura/spz.py "Validation / honest scope"):
//   aura write -> reference loadSpz: within quantization tolerance, degrees 0-3
//   reference saveSpz -> aura read:  within quantization tolerance
//   identical bytes, both decoders:  positions/scales/SH bit-exact; color/quat ~2e-7 (f32 vs f64)
//   129,531-carrier truck export (aura export-spz) -> reference loadSpz rc=0
//
// Minimal cross-validation harness around the ACTUAL nianticlabs/spz reference core.
// Modes:
//   pack   <in.bin> <out.spz>   read raw f32 arrays, saveSpz (default PackOptions, v4)
//   unpack <in.spz> <out.bin>   loadSpz (default UnpackOptions), dump raw f32 arrays
// Binary layout (little-endian): int32 numPoints, int32 shDegree, then f32 arrays in
// GaussianCloud field order: positions[N*3], scales[N*3], rotations[N*4],
// alphas[N], colors[N*3], sh[N*shDim*3].
#include "load-spz.h"
#include "splat-types.h"
#include <cstdio>
#include <cstdint>
#include <vector>
#include <fstream>
#include <string>

static int shDim(int deg){ switch(deg){case 0:return 0;case 1:return 3;case 2:return 8;case 3:return 15;case 4:return 24;} return 0; }

static std::vector<float> rd(std::ifstream&in,size_t n){ std::vector<float> v(n); in.read(reinterpret_cast<char*>(v.data()), n*sizeof(float)); return v; }
static void wr(std::ofstream&o,const std::vector<float>&v){ o.write(reinterpret_cast<const char*>(v.data()), v.size()*sizeof(float)); }

int main(int argc,char**argv){
  if(argc!=4){ fprintf(stderr,"usage: %s pack|unpack in out\n",argv[0]); return 2; }
  std::string mode=argv[1];
  if(mode=="pack"){
    std::ifstream in(argv[2],std::ios::binary);
    int32_t n=0,deg=0; in.read(reinterpret_cast<char*>(&n),4); in.read(reinterpret_cast<char*>(&deg),4);
    spz::GaussianCloud g; g.numPoints=n; g.shDegree=deg; g.antialiased=false;
    g.positions=rd(in,(size_t)n*3);
    g.scales   =rd(in,(size_t)n*3);
    g.rotations=rd(in,(size_t)n*4);
    g.alphas   =rd(in,(size_t)n);
    g.colors   =rd(in,(size_t)n*3);
    g.sh       =rd(in,(size_t)n*shDim(deg)*3);
    spz::PackOptions o; // defaults: version=4, from=UNSPECIFIED (=> identity conversion), sh bits 5/4
    if(!spz::saveSpz(g,o,std::string(argv[3]))){ fprintf(stderr,"saveSpz failed\n"); return 1; }
    return 0;
  } else if(mode=="unpack"){
    spz::UnpackOptions o; // to=UNSPECIFIED => identity
    spz::GaussianCloud g=spz::loadSpz(std::string(argv[2]),o);
    if(g.numPoints==0){ fprintf(stderr,"loadSpz failed/empty\n"); return 1; }
    std::ofstream out(argv[3],std::ios::binary);
    int32_t n=g.numPoints,deg=g.shDegree; out.write(reinterpret_cast<char*>(&n),4); out.write(reinterpret_cast<char*>(&deg),4);
    wr(out,g.positions); wr(out,g.scales); wr(out,g.rotations); wr(out,g.alphas); wr(out,g.colors); wr(out,g.sh);
    return 0;
  }
  fprintf(stderr,"bad mode\n"); return 2;
}
