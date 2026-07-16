struct sDecom;
#include <vector>
class cJustAList;
class cQuiver
{
    int xSize;//true number of verices
    std::vector<std::vector<int>> xArrows;
    cJustAList *xSeq;
public:
    cQuiver();
    ~cQuiver();
    void InitStar(int num_sub);
    int EulerForm(std::vector<int> &alpha, std::vector<int> &beta);
    int GenDecom(sDecom *d);
       void SetSize(int n) {xSize = n;}
    bool AddArrow(int i, int j);
    int Size() {return xSize;}
    int Arrow(int i, int j);
    bool AddVertex();
    bool DeleteArrowByVtx(int i, int j);
    void DeleteArrow(int i0, int k0);
    void DeleteVertex(int i);
    cJustAList *SinkSeq() {return xSeq;}
};

struct sDecom
{
    int size;
    std::vector<int> alpha;
    std::vector<std::vector<int>> smnds;
    std::vector<int> mltp;
    int n_smds;
    bool Expand(int * arr);
    sDecom(int size);
    sDecom(sDecom *p);
    int Solve(int * arr);
    void Permute(cJustAList *items);
};

class cJustAList
{
    int xData;
    cJustAList* xNext;
public:
    int Data() {return xData;}
    cJustAList* Next() {return xNext;}
    void SetNext(cJustAList* next) {xNext = next;}
    cJustAList(int data, cJustAList* next);
    ~cJustAList(); 
    cJustAList* Reverse();
    void SetData(int data) {xData = data;}
};

void my_assert(bool exp);
