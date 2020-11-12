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

// sa: 결과 접미사 배열 (원소의 값은 접미사의 시작 인덱스)
// pos: 그룹의 번호
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
        std::sort(sa.begin(), sa.end(), [&](int a, int b) {
            return cmp(a, b, d, N, pos);
        });

        std::vector<int> temp(N, 0);
        for (int i = 0; i < N - 1; i++)
            temp[i + 1] = temp[i] + cmp(sa[i], sa[i + 1], d, N, pos);
        for (int i = 0; i < N; i++)
            pos[sa[i]] = temp[i];

        if (temp[N - 1] == N - 1)
            break;
    }
}

// lcp: 접미사 배열에서 자신의 뒤에 있는 원소와의 공통된 prefix의 길이
void buildLCP(std::string& str, std::vector<int>& lcp)
{
    int N = (int)str.size();
    std::vector<int> sa, pos;
    buildSufArr(str, sa, pos);
    lcp.resize(N, 0);

    for(int i=0, k=0; i<N; i++, k=std::max(k-1, 0)) {
        if(pos[i] == N-1)
            continue;

        for(int j=sa[pos[i]+1]; str[i+k]==str[j+k]; k++);
        lcp[pos[i]] = k;
    }
}