#include <stack>
#include <algorithm>
#include <vector>

typedef long long Long;

Long findLargestFromHist(std::vector<Long>& hist)
{
    int n = hist.size();
    std::stack<std::pair<Long, int>> s;
    Long result = 0;
    s.emplace(hist[0], 0);
    for (int i = 1; i < n; ++i)
    {
        while (!s.empty() && hist[i] < s.top().first)
        {
            std::pair<Long, int> prev = s.top();
            s.pop();
            Long height = prev.first;
            int width = (s.empty() ? i : i - s.top().second - 1);
            result = std::max(width * height, result);
        }

        s.emplace(hist[i], i);
    }

    while (!s.empty())
    {
        std::pair<Long, int> prev = s.top();
        s.pop();
        Long height = prev.first;
        int width = (s.empty() ? n : n - s.top().second - 1);
        result = std::max(width * height, result);
    }

    return result;
}