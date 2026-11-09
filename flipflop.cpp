#include "matrix.h"
#include <cstdlib>  // For rand() and srand()
#include "quiver.h"
#include "ipca.h"
#include <cmath>
#include <limits>
#include <stdexcept>
/*
This file contains:
1) an imlementation of FlipFlop algorithm

2) a test function, TestIPCA, for FlipFlop, it samples random 
matrices of sizes given in a dimension vector of star quiver 
that is assumed to be proved for generic locally semi-simple; 
It means that MLE for iPCA exists generically, so random test works

3) Main function that iterates on all interesting dimension vectors
for a star quiver with 5 subspaces (meaning, 5 feature groups), checks,
using the quiver methods developed in quiver.h/.cpp whether a dimension
vector is generic locally semisimple, and if yes, calls TestIPCA.
The iterations stop after max_test_MLE times, if need more - just edit
this variable in Main

----NB: so far FlipFlop converges always, but in some cases the 
convergence is surprisngly slow, so we set a variable max_loop and 
in the test below, for some cases it didnt converge within limit;
---The results of each FlipFlop are printed out 
---If need a diffrent number of feature groups, not 5, edit variable
TESTSZ in Main, it should ne #feature_groups+1
---If having specific matrix of a good dimension vector (you can 
check it as in the function Main), pass it directly to FlipFlop
*/

std::vector<Matrix> FlipFlop(
    const std::vector<Matrix> &FeatureGroups,
    bool &converged,
    int &num_loop,
    int max_loop)
{
//the method obtains an MLE estimate for the covariance matrices of
//an iPCA model, such that for some observation vectors the features are groupes
//Thd model estimates the covariance of each feature in each group and of
//all samples, simultanously. MLE esists iff the corresponding representation
//of a star quiver is polystable. The user is expected to check the conditions
//on num_group (num of arrows in star quiver), num_samples (dimesnion of central node),
//and sizes of groups (dimensopns of side nodes) such that generic point is polystable
    if (FeatureGroups.empty())
        throw std::invalid_argument("at least one feature group is required");
    if (max_loop <= 0)
        throw std::invalid_argument("maximum iterations must be positive");
    int num_group = static_cast<int>(FeatureGroups.size());
    int num_samples = FeatureGroups[0].getRows();
    if (num_samples <= 0)
        throw std::invalid_argument("feature groups must contain at least one sample");
    long long total_features = 0;
    for (int grp = 0; grp < num_group; grp++)
    {
        if (FeatureGroups[grp].getRows() != num_samples)
             throw std::invalid_argument("different number of samples");
        if (FeatureGroups[grp].getCols() <= 0)
            throw std::invalid_argument("feature groups must contain at least one feature");
        total_features += FeatureGroups[grp].getCols();
        for (int i = 0; i < FeatureGroups[grp].getRows(); ++i)
            for (int j = 0; j < FeatureGroups[grp].getCols(); ++j)
                if (!std::isfinite(FeatureGroups[grp].at(i, j)))
                    throw std::invalid_argument("feature data must be finite");
    }
    if (total_features > std::numeric_limits<int>::max())
        throw std::invalid_argument("total number of features exceeds supported range");
    // checked that the input contains num_group matrices of feature groups per sample
    std::vector<Matrix> Covariance;
    Covariance.push_back(Matrix(num_samples,num_samples));
    for (int grp = 0; grp < num_group; grp++)
        Covariance.push_back(Matrix(FeatureGroups[grp].getCols(), FeatureGroups[grp].getCols()));
    for (int grp = 0; grp < num_group+1; grp++)
    {//set all covariance matices to identities of corresponding sizes
        int group_size = Covariance[grp].getCols();
        for (int i = 0; i< group_size; i++)
             Covariance[grp].set(i,i,1);

    }
    //now covariance is initialized and we start iterations
    
    int loop = 0;
    converged = false;
    num_loop = -1;
    double epsilon = 0;
    for (; loop < max_loop; loop++)
    {

        epsilon = 0;
        Matrix new_Sigma(num_samples,num_samples);//first update the
        //covariance among the samples
        for (int grp = 0; grp < num_group; grp++)
        {
            Matrix summand = FeatureGroups[grp];
            summand = summand.multiply(Covariance[grp+1].inverse());
            summand = summand.multiply(FeatureGroups[grp].transpose());
            new_Sigma.Add(summand);
        }
        new_Sigma.by_scalar(1.0/static_cast<double>(total_features));
        //check the magnitude of change
        Matrix diff = Covariance[0];
        diff.Add(new_Sigma,-1);
        epsilon += diff.Frobenius();
        Covariance[0] = new_Sigma; //set new value of Sigma
        //then update covariance for every feature group
        for (int grp = 0; grp < num_group; grp++)
        {
            Matrix new_Delta = FeatureGroups[grp].transpose();
            new_Delta = new_Delta.multiply(Covariance[0].inverse());
            new_Delta = new_Delta.multiply(FeatureGroups[grp]);
            new_Delta.by_scalar(1.0/num_samples);
            //check the magnitude of change
            Matrix diff = Covariance[grp+1];
            diff.Add(new_Delta,-1);
            epsilon += diff.Frobenius();
            Covariance[grp+1] = new_Delta;//set new value of Delta
        }
        if (epsilon < 1e-10)
        {
            converged = true;
            num_loop = loop + 1;
            break;//converged
        }
    }
    return Covariance;
}
bool TestIPCA(std::vector<int> &alpha)
{//input is a dimension vector for a star quiver such that 0 vertex is
//the center and all arrows are i->0
//the dimesion vector is supposed to be such that generic representation
//is polystable; so a representation is sampled at ramdom
    int n = alpha[0];//this is going to be the dimension at central node == #samples
//while alpha[1:size-1] are the sizes of feature  groups
    
//generate a random representation of star quiver of dimension 
//alpha == iPCA samples with 
//Feature groups of sizes alpha[1],...,alpha[size alpha - 1]
    std::vector<Matrix> Features;
    srand(17);//here user can change the seed
    for (int i = 1; i < alpha.size(); i++)
    {
        Matrix loc(n,alpha[i]);
        for (int j = 0; j < n; j++)
           for (int k = 0; k < alpha[i]; k++)
              loc.set(j,k, rand() % (9) + 1);
        Features.push_back(loc);
    }
    //the below call will result in convergence if and only if
    // (n;p) is generically polystable (a.k.a. locally semi-simple) dimension
    //and besides for generic matrices
    //if the condition not met, the process will diverge and at some point
    //covariance matrix will degenrate leading to exception in Matrix::Inverse
    //you can follow the convergence pace by watching epsilon variable line 206
    bool got_converged = false;
    int num_loop = -1, max_loop = 1000;
    std::vector<Matrix> Covariance = FlipFlop(Features, got_converged, num_loop, max_loop);
    if (got_converged)
    {
        std::cout << "FlipFlop coverged in " << num_loop << " steps. Sigma, covariance of samples:\n";
        Covariance[0].print();
        bool print_more = false;
        if (print_more)
        {
            for (int i = 0; i < alpha.size()-1; i++)
            {
                std::cout << "\n Delta[" << i+1 << "], covariance of group "<< i+1 << "\n";
                Covariance[i+1].print();
            }
        }
    }
    else
    {
        std::cout << "FlipFlop didn't coverge in " << max_loop << " steps.\n";
    }
    return got_converged;
}
bool Next(std::vector<int> &arr, int n)
{//given an array with n nonnegative elements
//to change array, passing to the next with the same sum
//if the last, return false
    int i = 0;
    for (;i < n; i++)
        if (arr[i] > 0)
            break;
    if (i == n-1)
        return false;
    arr[i+1]++;
    arr[0] = arr[i]-1;
    for (int j = 1; j <= i; j++)
        arr[j] = 0;
    return true;
}
int RunDemo()
{
    int TESTSZ = 6;//Test for 5 subspaces quiver having 6 vertices, you can edit as you need
    std::vector<int> al(TESTSZ,0);//this is going to iterate over different 
    //dimension vectors for star quiver, find dimension vectors
    //such that MLE exists generically, sample a representation of 
    //that dimension and apply FlipFlop to get MLE
    int num_test_MLE = 0, max_test_MLE = 20, max_total_dim = 100;
    for (int n = TESTSZ; n < max_total_dim; n++)//total 
    {//n = total dimension that may only be intersting > size        
        int i;
        al[0] = n;
        for (i = 1; i < TESTSZ; i++)
            al[i] = 0;
        while (Next(al,TESTSZ))
        {   
            int true_num_sub = 0, sum_sub=0;
            for (i = 1; i < TESTSZ; i++)
            {
                sum_sub += al[i];
                if (al[i] > al[i-1])
                    break;//not monotonously decreasing
                if (al[i]>0)
                    true_num_sub++;
            }
            if (i < TESTSZ)
                continue;//for non-duplicating, only consider decreasing vectors
            if (al[1]==al[0] || //degenerated subspace
                true_num_sub < 5 || //Euclidean case, less interesting
                2*sum_sub > true_num_sub*al[0] ||//reducible by castling
                sum_sub == al[0])//obviously locally semisimple
                continue;
            std::vector<int> group_sizes(al.begin() + 1, al.end());
            if (AnalyzeGenericMLEFeasibility(n, group_sizes).generic_mle_feasible)
            {
                num_test_MLE++;
                TestIPCA(al);
                if (num_test_MLE ==  max_test_MLE)
                    break;
            }
        }
        if (num_test_MLE ==  max_test_MLE)
            break;
    }
    return 0;
}