#include <string>
#include <vector>
#include <algorithm>

bool cmp(int i, int j, int d, int N, std::vector<int>& pos)
{
    if (pos[i] != pos[j])
        return pos[i] < pos[j];
        
    i += d;
    j += d;
    return (i < N && j < N) ? (pos[i] < pos[j]) : (i > j);
}

void buildSufArr(std::string& str, std::vector<int>& sa, std::vector<int>& pos)
{
    int N = (int)str.size();
    sa.resize(N, 0);
    pos.resize(N, 0);

    for (int i = 0; i < N; i++)
    {
        sa[i] = i;
        pos[i] = str[i];
    }
    
    for (int d = 1;; d *= 2)
    {
        std::sort(sa.begin(), sa.end(), cmp);

        std::vector<int> temp(N, 0);
        for (int i = 0; i < N - 1; i++)
            temp[i + 1] = temp[i] + cmp(sa[i], sa[i + 1], d, N, pos);
        for (int i = 0; i < N; i++)
            pos[sa[i]] = temp[i];

        if (temp[N - 1] == N - 1)
            break;
    }
}