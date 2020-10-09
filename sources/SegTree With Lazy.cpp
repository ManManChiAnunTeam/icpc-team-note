// 합을 저장하는 세그먼트 트리.

#include <vector>
#include <cmath>

template<typename T>
class SegTree
{
private:
    int count;
    std::vector<T> tree;
    std::vector<T> lazy;

    T initialize(int index, int start, int end, const std::vector<T>& original)
    {
        if (start == end)
            return tree[index] = original[start];
        
        int mid = (start + end) / 2;
        T left = initialize(index * 2, start, mid, original);
        T right = initialize(index * 2 + 1, mid + 1, end, original);

        return tree[index] = (left + right);
    }

    void propagate(int index, int start, int end)
    {
        if (lazy[index] != 0)
        {
            tree[index] += lazy[index] * (end - start + 1);

            if (start != end)
            {
                lazy[index * 2] += lazy[index];
                lazy[index * 2 + 1] += lazy[index];
            }

            lazy[index] = 0;
        }
    }

    T query(int index, int nodeStart, int nodeEnd, int reqStart, int reqEnd)
    {
        propagate(index, nodeStart, nodeEnd);

        int nodeMid = (nodeStart + nodeEnd) / 2;

        if (nodeStart > reqEnd || nodeEnd < reqStart)
            return 0;

        else if (nodeStart >= reqStart && nodeEnd <= reqEnd)
            return tree[index];
        
        else
        {
            T left = query(index * 2, nodeStart, nodeMid, reqStart, reqEnd);
            T right = query(index * 2 + 1, nodeMid + 1, nodeEnd, reqStart, reqEnd);
            return left + right;
        }
    }

    void update(T add, int dataStart, int dataEnd, int treeIndex, int treeStart, int treeEnd)
    {
        propagate(treeIndex, treeStart, treeEnd);

        int treeMid = (treeStart + treeEnd) / 2;

        if (dataEnd < treeStart || treeEnd < dataStart)
            return;
        
        if (dataStart <= treeStart && treeEnd <= dataEnd)
        {
            tree[treeIndex] += add * (treeEnd - treeStart + 1);
            if (treeStart != treeEnd)
            {
                lazy[treeIndex * 2] += add;
                lazy[treeIndex * 2 + 1] += add;
            }
            return;
        }

        update(add, dataStart, dataEnd, treeIndex * 2, treeStart, treeMid);
        update(add, dataStart, dataEnd, treeIndex * 2 + 1, treeMid + 1, treeEnd);

        tree[treeIndex] = tree[treeIndex * 2] + tree[treeIndex * 2 + 1];
    }
    
public:
    SegTree(const std::vector<T>& original)
    {
        count = (int)original.size();
        int treeHeight = (int)ceil(log2(count));
        int vecSize = (1 << (treeHeight + 1));
        tree.resize(vecSize);
        lazy.resize(vecSize);
        initialize(1, 0, count - 1, original);
    }

    SegTree(int size)
    {
        count = size;
        int treeHeight = (int)ceil(log2(count));
        int vecSize = (1 << (treeHeight + 1));
        tree.resize(vecSize);
        lazy.resize(vecSize);
    }

    T query(int start, int end)
    {
        return query(1, 0, count - 1, start, end);
    }

    void update(T add, int start, int end)
    {
        update(add, start, end, 1, 0, count - 1);
    }
};