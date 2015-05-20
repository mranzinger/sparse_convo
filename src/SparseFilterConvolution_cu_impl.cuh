
class lua_State;

namespace sparse_convo { namespace cuda {

class SparseFilterConvo
{
public:
    static int UpdateOutput(lua_State *L);
    static int UpdateGradInput(lua_State *L);
    static int AccGradParameters(lua_State *L);
};

} }
