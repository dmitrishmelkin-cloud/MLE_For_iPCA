
#include "quiver.h"

void my_assert(bool exp)
{
    if (!exp)
    {
        bool dummy  = exp;//to put a breakpoint
    }
}

cQuiver::cQuiver()
{
    xSize = 0;
    xSeq = 0;
}
cQuiver::~cQuiver()
{
    delete xSeq;
}
int cQuiver::EulerForm(std::vector<int> &alpha, std::vector<int> &beta)
{
    int eu = 0, i = 0;
    for (i = 0; i < xSize; i++)
        eu += alpha[i]*beta[i];

    for (i = 0; i < xSize; i++)
    {
        for (int j = 0; j < xArrows[i].size(); j++)
        {
            int k = xArrows[i][j];
            if (k == -1)
                break;
            eu -= alpha[i]*beta[k];
        }
    }
    return eu;
}

void cQuiver::InitStar(int num_sub)
{
    if (num_sub <= 0)
        return;
    xSize = num_sub + 1;
    for (int i = 0; i < xSize; i++)
    {
        std::vector<int> loc;
        xArrows.push_back(loc);
        if (i)
        {
            xArrows[i].push_back(0);
            xSeq = new cJustAList(i,xSeq);
        }
    }
    xSeq = new cJustAList(0,xSeq);
}

void TwoVtxCanon(int x, int y, int r, 
                 int &x1, int &y1, int &m1,
                 int &x2, int &y2, int &m2)
{//the quiver has 2 vertices, r arrows 0 to 1, and dimension (x,y)
//for this case the generic (canonical) decompostion is possible
//to write down
    if (r < 0)
        return;
    if (y == 0)
    {
        x1 = 1; y1 = 0; m1 = x;
        x2 = 0; y2 = 0; m2 = 0;
    }
    if (x == 0)
    {
        x1 = 0; y1 = 1; m1 = y;
        x2 = 0; y2 = 0; m2 = 0;
    }
    else if (r == 0)
    {
        x1 = 1; y1 = 0; m1 = x;
        x2 = 0; y2 = 1; m2 = y;
    }
    else if (r == 1)
    {
        if (x == y)
        {
            x1 = 1; y1 = 1; m1 = x;
            x2 = 0; y2 = 0; m2 = 0;
        }
        else if (x > y)
        {
            x1 = 1; y1 = 0; m1 = x - y;
            x2 = 1; y2 = 1; m2 = y;
        }
        else
        {    
            x1 = 1; y1 = 1; m1 = x;
            x2 = 0; y2 = 1; m2 = y - x;
        }
    }
    else if (x == y && r == 2)
    {
        x1 = 1; y1 = 1; m1 = x;
        x2 = 0; y2 = 0; m2 = 0;
    }
    else
    {
        double a = ((double)x)/y;
        int u1,u2,v1,v2;
        if (r >= 2*a && (r-2*a)*(r-2*a) > r*r - 4)
        {
            u1 = 0, v1 = 1, u2 = 1, v2 = r;
            double val1 = 0., val2 = 1./r;
            while (a > val2)
            {
                int u3 = -u1 +r*v1, v3 = -r*u1+(r*r-1)*v1;
                double val3 = ((double)u3)/v3;
                u1 = u2; v1 = v2; val1 = val2;
                u2 = u3; v2 = v3; val2 = val3;
            }
        }
        else if (r <= 2*a && (r-2*a)*(r-2*a) > r*r - 4)
        {
            u1 = 1, v1 = 0, u2 = r, v2 = 1;
            double val1 = 0., val2 = r;
            while (a < val2)
            {
                int u3 = (r*r-1)*u1 - r*v1, v3 = r*u1 - v1;
                double val3 = ((double)u3)/v3;
                u1 = u2; v1 = v2; val1 = val2;
                u2 = u3; v2 = v3; val2 = val3;
            }
        }
        else
        {//imaginary root
            x1 = x; y1 = y; m1 = 1;
            x2 = 0; y2 = 0; m2 = 0;
            return;
        }
        int eu12 = u1*u2+v1*v2-r*u1*v2;
        int eu21 = u1*u2+v1*v2-r*u2*v1;
        my_assert(eu12 == 0 || eu21 == 0);;
        if (eu12 == 0)
        {
            x1 = u1; y1 = v1; x2 = u2; y2 = v2;
        }
        else if (eu21 == 0)
        {
            x1 = u2; y1 = v2; x2 = u1; y2 = v1;
        }
//find multiplicities by solving the linear system
        int det = x1*y2 - x2*y1;
        my_assert(det != 0);

        int d1 = x*y2 - x2*y;
        int d2 = x1*y - x*y1;
        m1 = d1/det; m2 = d2/det;
        my_assert(m1 >= 0 && m2 >= 0 && m1*det == d1 && m2*det == d2);

        if (m1 == 0)
        {
            x1 = x2; y1 = y2; m1 = m2; x2 = y2 = m2 = 0;
        }
    }
}

int cQuiver::GenDecom(sDecom *d)
{//Derksen-Weyman algorithm
//set starting decomp, firstly assuming that all arrows are from grt to less
    d->n_smds = 0;
    my_assert(d->alpha.size()>0);//the dimension is set before calling
    if (!xSeq)//xSeq is a topological sorting that may be pre-defined, or
    {//if not defined, we calculate a sequence of sinks
        cJustAList *tail = 0;
        std::vector<int> col(xSize,0);
        for (int i = 0; i < xSize; i++)
            col[i] = 0;//white then black
        for (int cnt = 0; cnt < xSize; cnt++)
        {
//find a vertex that is not yet in xSeq and all arrows go to vertices in xSeq
            int i = 0, j=0;
            for (; i < xSize; i++)
            {//first check that this is new
                if (col[i])
                    continue;//old
                for (j = 0; j < xArrows[i].size(); j++)
                {
                     if (!col[xArrows[i][j]])
                        break;
                }
                if (j == xArrows[i].size())
                    break;//all arrows are to black vtices
            }
            if (i == xSize){
                my_assert(0);//all white vertices have arrows to white vertices
                return -1;}//oriented cycle exists
            if (!xSeq){
                xSeq = new cJustAList(i,0); tail = xSeq;}
            else{
                tail->SetNext(new cJustAList(i,0)); tail = tail->Next();}
            col[i] = 1;
        }
    }
    for (cJustAList *jal = xSeq; jal; jal = jal->Next())
    {
        int i = jal->Data();
        if (d->alpha[i] == 0)
            continue;
        std::vector<int> loc;
        for (int j = 0; j < xSize; j++)
        {
            if (j == i)
                loc.push_back(1);
            else
                loc.push_back(0);
        }
        d->smnds.push_back(loc);
        d->mltp.push_back(d->alpha[i]);
        d->n_smds++;
    }
    while (d->n_smds >1)
    {
        int clm = -1, clm2 = -1, i = 0;
        for (i = 0; i < d->n_smds; i++)
        {   
            for (int j = i+1; j < d->n_smds; j++)
            {
                int eu = EulerForm(d->smnds[i],d->smnds[j]);
                my_assert(eu == 0);
                int eu2 = EulerForm(d->smnds[j],d->smnds[i]);
                if (eu2 < 0 && (clm < 0 || clm2 - clm > j - i))
                {
                    clm = i; clm2 = j;
                }
            }
        }
        if (clm < 0)
            break;//reached generic decomposition
        if (clm2-clm > 1)
        {//make a permutation           
            cJustAList *perm = new cJustAList(clm2,0), *p2 = perm; 
            perm = new cJustAList(clm,perm);
            cJustAList *p1 = perm; 
            cJustAList *tail = p2, *prv = 0;
            for (i = clm + 1; i < clm2; i++)
            {  
                int eu = EulerForm(d->smnds[clm2],d->smnds[i]);
                if (eu == 0)
                {
                    tail->SetNext(new cJustAList(i,0));
                    tail = tail->Next();
                }
                else
                {
                    if (!prv)
                    {
                        prv = new cJustAList(i,p1);
                        perm = prv;
                    }
                    else
                    {
                        prv->SetNext(new cJustAList(i,p1));
                        prv = prv->Next();
                    }
                    clm++;
                }
            }
            d->Permute(perm);
            delete perm;
            clm2 = clm + 1;
        }
        int c = clm;
        int qc = EulerForm(d->smnds[c],d->smnds[c]);
        int qcn = EulerForm(d->smnds[c+1],d->smnds[c+1]);
        int r = EulerForm(d->smnds[c+1],d->smnds[c]);
        int p = d->mltp[c], q = d->mltp[c+1];
        int x1 = -1, y1 = -1, m1 = -1, x2 = -1, y2 = -1, m2 = -1;
        if (qc == 1 && qcn == 1)
        {//both are real
            TwoVtxCanon(q,p,-r,x1,y1,m1,x2,y2,m2);
        }
        else if (qc == 1 && qcn <= 0)
        {
            if (p + q*r >= 0)
            {
                x1 = 1; y1 = -r; m1 = q;
                x2 = 0; y2 = p + q*r ? 1 : 0; m2 = p + q*r;
            }
            else
            {
                x1 = q; y1 = p; m1 = 1;
                x2 = 0; y2 = 0; m2 = 0;
            }
        }
        else if (qc <= 0 && qcn == 1)
        {
            if (q + p*r > 0)
            {
                x1 = 1; y1 = 0; m1 = q + p*r;
                x2 = -r; y2 = 1; m2 = p;
            }
            else if (q + p*r == 0)
            {
                x1 = -r; y1 = 1; m1 = p;
                x2 = 0; y2 = 0; m2 = 0;
            }
            else
            {
                x1 = q; y1 = p; m1 = 1;
                x2 = 0; y2 = 0; m2 = 0;
            }
        }
        else
        {
            my_assert(qc <=0 && qcn <= 0);
            x1 = q; y1 = p; m1 = 1;
            x2 = 0; y2 = 0; m2 = 0;
        }
//now replace smnds[c] and (if m2) smnds[c+1] with their linear combination
        std::vector<int> u,v;//new combinations
        for (int i = 0; i < xSize; i++)
        {
            u.push_back(x1*d->smnds[c+1][i]+y1*d->smnds[c][i]);
            v.push_back(x2*d->smnds[c+1][i]+y2*d->smnds[c][i]);
        }
//saved new summands to u and v, can replace
        for (int i = 0; i < xSize; i++)
        {
            d->smnds[c][i] = u[i];
            if (m2)
               d->smnds[c+1][i] = v[i];
        }
        d->mltp[c] = m1;
        if (m2)
           d->mltp[c+1] = m2;
        else//delete a summand
        {
            d->smnds.erase(d->smnds.begin()+c+1);
            d->mltp.erase(d->mltp.begin()+c+1);
            d->n_smds--;
        }
        
    }
//#ifdef _DEBUG
//check the results
    int ii;
    for (ii = 0; ii < xSize; ii++)
    {
        int a = d->alpha[ii];
        for (int j = 0; j < d->n_smds; j++)
            a -= d->smnds[j][ii]*d->mltp[j];
        my_assert(a == 0);//check that it is a decomposition
    }
    for (ii = 0; ii < d->n_smds; ii++)
    {
        for (int j = ii; j < d->n_smds; j++)
        {
            int forw = EulerForm(d->smnds[ii],d->smnds[j]);
            int back = EulerForm(d->smnds[j],d->smnds[ii]);
            if (ii == j)
                my_assert(forw <= 1);//a root
            else
                my_assert(forw == 0 && back >= 0);
        }
    }
//#endif _DEBUG;
    return d->n_smds;
}

bool sDecom::Expand(int * arr)
{//check that ar is a non-negative linear comb of smnds, 
//write coef to mtp, do it recursively
    int stat = Solve(arr);
    if (stat == 1)
        return true;
    if (stat != -1)
        return false;
    for (int i = 0; i < n_smds; i++)
    {
//compare alpha and smnds[i]
        int j = 0;
        int grt = -1;
        for (; j < size; j++)
        {
            if (smnds[i][j] > arr[j])
                break;
            if (smnds[i][j] < arr[j])
                grt = j;
        }
        if (j < size)
            continue;
        for (j = 0; j < size; j++)
            arr[j] -= smnds[i][j];
        mltp[i]++;
        if (grt == -1)
            return true;//alpha is now 0
        if (Expand(arr))
            return true;
        for (j = 0; j < size; j++)
            arr[j] += smnds[i][j];
        mltp[i]--;
    }
    return false;
}


sDecom::sDecom(int n)
{
    size = n;
    n_smds = 0;
    //filling in containers is to be done by user of this constructor
    
}
cJustAList::cJustAList(int data, cJustAList* next)
{
    xData = data;
    xNext = next;
}

cJustAList::~cJustAList()
{
    while( xNext )
    {
        cJustAList * tmp = xNext;
        xNext = tmp -> xNext;
        tmp -> xNext = 0;
        delete tmp;
    }
}
cJustAList *cJustAList :: Reverse()
{
    cJustAList *prev    = 0;
    cJustAList *current = this;
    while( current )
    {
        cJustAList *next = current->xNext;
        current->xNext = prev;
        prev = current;
        current = next;
    }
    return prev;
} // END cJustAList::Reverse()

bool cQuiver::AddArrow(int i, int j)
{
    if (i == j)
        return false;//loops not supported
//check no arrow goes reversely 
    bool rev_ar = false;
    for(int k = 0; k < xArrows[j].size(); k++)
    {
        int l = xArrows[j][k];
        if (l < 0)
            break;
        if (l == i)
        {
            rev_ar = true;
            break;
        }
    }
    if (rev_ar)
        return false;
//check that new arrow does not yield oriented cycles 
    int state = 0;
    for (cJustAList *jal = xSeq; jal; jal = jal->Next())
    {
        int k = jal->Data();
        if (k == j)
            state = 1;
        else if (k == i)
        {
            if (!state)
                state = -1;//i before j
            break;
        }
    }
    if (state == -1)
    {
        cJustAList *sqn = 0, *tail = 0;
        std::vector<int> col(xSize,0);
        int ii = 0, jj=0;
        
        for (int cnt = 0; cnt < xSize; cnt++)
        {
            //find a vertex that is not yet in sqn and all arrows go to vertices in sqn
            for (ii = 0; ii < xSize; ii++)
            {//first check that this is new
                if (col[ii])
                    continue;//old
                if (ii == i && !col[j])
                    continue;//i will have an arrow to j
                for (int jj = 0; jj < xArrows[ii].size(); jj++)
                {
                    if (xArrows[ii][jj] == -1)
                        break;
                    if (!col[xArrows[ii][jj]])
                        break;
                }
                if (xArrows[ii][jj] == -1)
                    break;//all arrows are to black vtices
            }
            if (ii == xSize)
            {//all white vertices have arrows to white vertices
                delete sqn;
                sqn = 0;
                break;
            }
            if (!sqn){
                sqn = new cJustAList(ii,0); tail = sqn;}
            else{
                tail->SetNext(new cJustAList(ii,0)); tail = tail->Next();}
            col[ii] = 1;
        }
        if (!sqn)
            return false;//oriented cycles
        delete xSeq;
        xSeq = sqn;
    }
//end check
    xArrows[i].push_back(j);
     return true;
}

sDecom::sDecom(sDecom *p)
{
    n_smds = p->n_smds;
    size = p->size;
    for (int i = 0; i < size; i++)
        alpha.push_back(p->alpha[i]);
 
     for (int j = 0; j < n_smds; j++)
     {
        mltp.push_back(p->mltp[j]);
        std::vector<int> loc;
        for (int i = 0; i < size; i++)
            loc.push_back(p->smnds[j][i]);
        smnds.push_back(loc);
     }
}

inline double mod(double a)
{
    return a > 0 ? a : -a;
}
int sDecom::Solve(int * arr)
{//solve linear system defined by coef from smnds
//and by right terms from arr, write coef to mltp
//return
// -1 if smnds are linear dependent
//  0 if no solution
//  1 if unique solution is nin-negative integer
//  2 if unique solution is either negative or not integer
    std::vector<std::vector<double>> coef;
    std::vector<double> right;
    std::vector<double> sol;
//initialize
    int i = 0;
    for (i = 0; i < n_smds; i++)
    {
        sol[i] = 0.;
        std::vector<double> smd_copy;
        for (int j = 0; j < size; j++)
        {
            smd_copy.push_back(smnds[i][j]);
            if (i == 0)
                right.push_back(arr[j]);
        }
        coef.push_back(smd_copy);
    }
//take next column
    for (i = 0; i < n_smds; i++)
    {
        double least = 0.;
        int best = -1;
        int j = i;
        for (j = i; j < size; j++)
        {
            if (coef[i][j] == 0)
                continue;
            if (best != -1 && mod(coef[i][j]) >= least)
                continue;
            least = mod(coef[i][j]);
            best = j;
        }
        if (best == -1)
            return -1;//linear dependent
        if (best > i)
        {//permute lines i and best
            double tmp;
            for (j = i; j < n_smds; j++)
            {
                tmp = coef[j][i];
                coef[j][i] = coef[j][best];
                coef[j][best] = tmp;
            }
            tmp = right[i];
            right[i] = right[best];
            right[best] = tmp;
        }
        for (j = i+1; j < size; j++)
        {//subtract line[i]/mul from line j
            double mul = coef[i][j]/coef[i][i];
            coef[i][j] = 0.;
            for (int k = i+1; k < n_smds; k++)
                coef[k][j] -= coef[k][i]*mul;
            right[j] -= right[i]*mul;
        }
    }
//check that a solution exists; define the presision
    double eps = 0.01;
    for (i = n_smds + 1; i < size; i++)
    {
        if (mod(right[i]) > eps)
            break;
    }
    if (i < size)
        return 0;//no solution
    for (i = n_smds - 1; i >= 0; i--)
    {
        sol[i] = right[i];
        for (int j = i+1; j < n_smds; j++)
            sol[i] -= coef[j][i]*sol[j];
        my_assert(mod(coef[i][i]) > eps);
        sol[i] /= coef[i][i];
    }
//check that the solution is natural
    for (i = 0; i < n_smds; i++)
    {
        if (sol[i] < -eps)
            break;
        int s = (int)(sol[i]+eps);
        if (mod(sol[i] - s) > eps)
            break;
    }
    if (i < n_smds)
        return 2;
    for (i = 0; i < n_smds; i++)
        mltp[i] = (int)(sol[i]+eps);
//#ifdef _DEBUG
//check that solution
    for (i = 0; i < size; i++)
    {
        int a = arr[i];
        for (int j = 0; j < n_smds; j++)
            a -= smnds[j][i]*mltp[j];
        my_assert(a == 0);//check that it is a decomposition
    }
//#endif _DEBUG
    return 1;
}

int cQuiver::Arrow(int i, int j)
{
    if (i < 0 || i >= xSize || j < 0)
        return -1; 
    return xArrows[i][j];
}

bool cQuiver::AddVertex()
{
    xSeq = new cJustAList(xSize,xSeq);
    xSize++;
    return true;
}


void cQuiver::DeleteArrow(int i0, int k0)
{
    if (i0 < 0 || i0 >= xSize || k0 < 0 || k0 > xArrows[i0].size())
        return;
    xArrows[i0].erase(xArrows[i0].begin()+ k0);
    
}
void cQuiver::DeleteVertex(int i)
{
    if (i >= xSize)
        return;
//first remove all arrows to i
    int j = 0;
    for (; j < xSize; j++)
    {
        for (int k = 0; k < xArrows[j].size(); k++)
        {
            if (xArrows[j][k] < 0)
                break;
            if (xArrows[j][k] == i)
                DeleteArrow(j,k);
        }
    }
//now remove all in i
    
    xArrows[i].clear();
    xArrows.erase(xArrows.begin()+i);
    xSize--;
//remove i from the sequence and replace l (>i) by l-1
    cJustAList *prv = 0;
    for (cJustAList *jal = xSeq; jal;)
    {
        int j = jal->Data();
        if (j == i)
        {
            cJustAList *nxt = jal->Next();
            jal->SetNext(0);
            delete jal;
            if (!prv)
                xSeq = nxt;
            else
                prv->SetNext(nxt);
            jal = nxt;
        }
        else
        {
            if (j > i)
                jal->SetData(j-1);
            prv = jal; jal = jal->Next();
        }
    }
}

void sDecom::Permute(cJustAList *items)
{
    if (!items)
        return;
//items should be a permuted conseq sequence of integers
//between 0 and n_smnds; we need to copy those
//summands and their multiplicities and then write
//each to the correct place
    std::vector<std::vector<int>> summands;
    std::vector<int> multi;
    int i = 0;
    for (; i < size; i++)
        multi.push_back(-1);
    int max = -1, min = size, num = 0;
    cJustAList *jal = items;
    for (; jal; jal = jal->Next())
    {
        num++;
        int j = jal->Data();
        if (multi[j] != -1){//twice i in the list
            my_assert(!"bad data");return;}
        if (j < min)
            min = j;
        if (j > max)
            max = j;
        multi[j] = mltp[j];
        for (int k = 0; k < size; k++)
            summands[j][k] = smnds[j][k];
    }
    if (max<0 || max>size-1 || min<0 || max>size-1 || max-min != num-1){
        my_assert(!"bad data");return;}
    i = min;
    for (jal = items; jal; jal = jal->Next())
    {
        int j = jal->Data();
        mltp[i] = multi[j];
        for (int k = 0; k < size; k++)
            smnds[i][k] = summands[j][k];
        i++;
    }
}


