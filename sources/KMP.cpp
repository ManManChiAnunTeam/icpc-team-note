#include <string>
#include <vector>

std::vector<int> getFail(std::string& str)
{
    std::vector<int> fail(str.size(), 0);
    for (int i = 1, j = 0; i < (int)str.size(); ++i)
    {
        while (j > 0 && str[i] != str[j])
            j = fail[j-1];
        if (str[i] == str[j])
            fail[i] = ++j;
    }

    return fail;
}

void KMP(std::string& para, std::string& target, std::vector<int>& fail, std::vector<int>& found)
{
    fail = getFail(target);
    found.clear();

    for (int i = 0, j = 0; i < (int)para.size(); ++i)
    {
        while (j > 0 && para[i] != target[j])
            j = fail[j-1];
        if (para[i] == target[j])
        {
            if (j == (int)target.size()-1)
            {
                found.push_back(i-target.size()+2);
                j = fail[j];
            }
            else j++;
        }
    }
}