# 2020 ACM-ICPC ManManChiAnunTeam Team Note

## 꿀팁
- 우리에게 불가능이라는 선택지는 없다.
- DP가 아닌 것 같은 문제도 DP일 수 있다.
- 그리디가 아닌 것 같은 문제도 그리디일 수 있다.
- 침착하자
- 테스트를 많이 해보자
- 쉬운 거 먼저 풀자
- long long
- long double
- n, m 구분

## 나올 수 있는 알고리즘들
- DP on trees

## Tree

### LCA
```C++
// O(nlogn)
const int MAX_D = 17;
const int MAXN = 100000;

int N, p[MAXN][MAX_D], depth[MAXN];
vector<int> adj[MAXN];
bool vst[MAXN];

void dfs(int i, int d) {
    vst[i] = true, depth[i] = d;
    for (int j : adj[i])
        if (!vst[j]) {
            p[j][0] = i;
            dfs(j, d + 1);
        }
}

void construct_lca() {
    dfs(0, 0);
    for (int j = 1; j < MAX_D; j++)
        for (int i = 1; i < N; i++)
            p[i][j] = p[p[i][j - 1]][j - 1];
}

int find_lca(int a, int b) {
    // Make a have a higher depth
    if (depth[a] < depth[b]) swap(a, b);

    // Elevate a to the depth of b
    int depth_diff = depth[a] - depth[b];
    for (int j = MAX_D - 1; j >= 0; j--)
        if (depth_diff & (1 << j))
            a = p[a][j];

    if (a == b) return a;

    for (int j = MAX_D - 1; j >= 0; j--)
        if (p[a][j] != p[b][j])
            a = p[a][j], b = p[b][j];

    return p[a][0];
}

int main() {
    cin >> N;
    for (int i = 0; i < N - 1; i++) {
        int x, y;
        cin >> x >> y;
        x--; y--;
        adj[x].push_back(y), adj[y].push_back(x);
    }
    construct_lca();
}
```

### Segment tree
```C++
template<typename T> class SegTree {
    // Segment tree가 구하고자 하는 값에 따라 변경
    const T BOUND = 0;
    T f(T a, T b) { return a + b; }

    int MIN, MAX;
    vector<T> node;

    T query(int k, int node_s, int node_e, int req_s, int req_e) {
        if (req_e < node_s || node_e < req_s)
            return BOUND;
        if (req_s <= node_s && node_e <= req_e)
            return node[k];
        int node_m = (node_s + node_e) / 2;
        T left = query(k * 2, node_s, node_m, req_s, req_e);
        T right = query(k * 2 + 1, node_m + 1, node_e, req_s, req_e);
        return f(left, right);
    }
    void update(int k, int node_s, int node_e, int req_i, T value) {
        if (req_i < node_s || node_e < req_i)
            return;
        if (node_s == node_e) {
            node[k] = value;
            return;
        }
        int node_m = (node_s + node_e) / 2;
        update(k * 2, node_s, node_m, req_i, value);
        update(k * 2 + 1, node_m + 1, node_e, req_i, value);
        node[k] = f(node[k * 2], node[k * 2 + 1]);
    }
public:
    SegTree(int MIN, int MAX) : MIN(MIN), MAX(MAX) { node.resize(4 * (MAX - MIN)); }
    T query(int start, int end) { return query(1, MIN, MAX, start, end); }
    void update(int i, T value) { update(1, MIN, MAX, i, value); }
};
```

### Dynamic segment tree
```C++
template<typename T> class SegTree {
    // Segment tree가 구하고자 하는 값에 따라 변경
    const T BOUND = 0;
    inline T f(T a, T b) { return a + b; }

    struct Node {
        int s, e;
        T value;
        int l, r;
        Node(int s, int e, T value, int l, int r): s(s), e(e), value(value), l(l), r(r) {}
    };
    vector<Node> node;
    int MIN, MAX;

    T query(int k, int req_s, int req_e) {
        if(req_e < node[k].s || node[k].e < req_s)
            return BOUND;
        if(req_s <= node[k].s && node[k].e <= req_e)
            return node[k].value;
        T left = node[k].l != -1 ? query(node[k].l, req_s, req_e) : BOUND;
        T right = node[k].r != -1 ? query(node[k].r, req_s, req_e) : BOUND;
        return f(left, right);
    }
    void update(int k, int req_i, T value) {
        if (node[k].s == node[k].e) {
            node[k].value = value;
            return;
        }
        int m = (node[k].s + node[k].e) / 2;
        if(req_i <= m) {
            if(node[k].l == -1) {
                node[k].l = node.size();
                node.emplace_back(node[k].s, m, BOUND, -1, -1);
            }
            update(node[k].l, req_i, value);
        } else {
            if(node[k].r == -1) {
                node[k].r = node.size();
                node.emplace_back(m + 1, node[k].e, BOUND, -1, -1);
            }
            update(node[k].r, req_i, value);
        }
        T left = node[k].l != -1 ? node[node[k].l].value : BOUND;
        T right = node[k].r != -1 ? node[node[k].r].value : BOUND;
        node[k].value = f(left, right);
    }
public:
    SegTree(int MIN, int MAX) : MIN(MIN), MAX(MAX) { node.emplace_back(MIN, MAX, BOUND, -1, -1); }
    T query(int start, int end) { return query(0, start, end); }
    void update(int i, T value) { update(0, i, value); }
};
```

### Segment tree with lazy propagation
```C++
template<typename T> class SegTree {
    int MIN, MAX;
    vector<T> node, lazy;

    T init(int k, int node_s, int node_e, const vector<T>& data) {
        if (node_s == node_e)
            return node[k] = data[node_s];
        int node_m = (node_s + node_e) / 2;
        T left = init(k * 2, node_s, node_m, data);
        T right = init(k * 2 + 1, node_m + 1, node_e, data);
        return node[k] = left + right;
    }
    void prop(int k, int node_s, int node_e) {
        node[k] += lazy[k] * (node_e - node_s + 1);
        if (node_s != node_e) {
            lazy[k * 2] += lazy[k];
            lazy[k * 2 + 1] += lazy[k];
        }
        lazy[k] = 0;
    }
    T query(int k, int node_s, int node_e, int req_s, int req_e) {
        prop(k, node_s, node_e);
        if (req_e < node_s || node_e < req_s)
            return 0;
        if (req_s <= node_s && node_e <= req_e)
            return node[k];
        int node_m = (node_s + node_e) / 2;
        T left = query(k * 2, node_s, node_m, req_s, req_e);
        T right = query(k * 2 + 1, node_m + 1, node_e, req_s, req_e);
        return left + right;
    }
    void update(int k, int node_s, int node_e, int req_s, int req_e, T add) {
        prop(k, node_s, node_e);
        if (req_e < node_s || node_e < req_s)
            return;
        if (req_s <= node_s && node_e <= req_e) {
            lazy[k] += add;
            prop(k, node_s, node_e);
            return;
        }
        int node_m = (node_s + node_e) / 2;
        update(k * 2, node_s, node_m, req_s, req_e, add);
        update(k * 2 + 1, node_m + 1, node_e, req_s, req_e, add);
        node[k] = node[k * 2] + node[k * 2 + 1];
    }
public:
    SegTree(int MIN, int MAX) : MIN(MIN), MAX(MAX) {
        node.resize(4 * (MAX - MIN));
        lazy.resize(4 * (MAX - MIN));
    }
    SegTree(const vector<T>& data) : MIN(0), MAX(int(data.size()) - 1) {
        node.resize(4 * (MAX - MIN));
        lazy.resize(4 * (MAX - MIN));
        init(1, MIN, MAX, data);
    }
    T query(int start, int end) { return query(1, MIN, MAX, start, end); }
    void update(int start, int end, T add) { update(1, MIN, MAX, start, end, add); }
};
```

### Persistent Segment Tree
```C++
class PST {
    struct Node {
        int left, right;  // [left, right]
        int sum;
        Node *lchild, *rchild;

        Node(int left, int right) : left(left), right(right), sum(0), lchild(nullptr), rchild(nullptr) {}
    };

    Node *root[MAXN + 1];  // root[x]: tree of 0 ~ x-1
    vector<Node *> node_ptrs;

    Node *update_(Node *this_node, int y, bool is_new) {
        int left = this_node->left;
        int right = this_node->right;
        int mid = (left + right) / 2;

        Node *new_node;
        if (!is_new) {
            new_node = new Node(left, right);
            node_ptrs.push_back(new_node);
            new_node->lchild = this_node->lchild;
            new_node->rchild = this_node->rchild;
        } else
            new_node = this_node;

        // Leaf node
        if (left == right) {
            new_node->sum = this_node->sum + 1;
            return new_node;
        }

        if (y <= mid) {  // Left
            if (!new_node->lchild) {
                new_node->lchild = new Node(left, mid);
                node_ptrs.push_back(new_node->lchild);
                update_(new_node->lchild, y, true);
            } else
                new_node->lchild = update_(new_node->lchild, y, false);
        } else {  // Right
            if (!new_node->rchild) {
                new_node->rchild = new Node(mid + 1, right);
                node_ptrs.push_back(new_node->rchild);
                update_(new_node->rchild, y, true);
            } else
                new_node->rchild = update_(new_node->rchild, y, false);
        }

        int sum = 0;
        if (new_node->lchild) sum += new_node->lchild->sum;
        if (new_node->rchild) sum += new_node->rchild->sum;
        new_node->sum = sum;
        return new_node;
    }

    int get_sum_(Node *here, int b, int t) {
        if (!here || t < here->left || here->right < b)
            return 0;
        else if (b <= here->left && here->right <= t)
            return here->sum;
        else
            return get_sum_(here->lchild, b, t) + get_sum_(here->rchild, b, t);
    }

public:
    PST() {
        root[0] = new Node(0, MAXY);
        node_ptrs.push_back(root[0]);
        for (int i = 1; i <= MAXN; i++) root[i] = nullptr;
    }

    void update(int xi, int y) {
        if (!root[xi + 1])
            root[xi + 1] = update_(root[xi], y, false);
        else
            update_(root[xi + 1], y, true);
    }

    // Sum of 0 ~ x-1
    int get_sum(int xi, int b, int t) {
        return get_sum_(root[xi + 1], b, t);
    }

    ~PST() {
        for (Node *p : node_ptrs) delete p;
    }
};
```

### Li-Chao Tree
```C++
struct LiChaoTree {
    struct Line {
        ll a, b;
        ll f(ll x) {
            return a * x + b;
        }
    };

    struct Node {
        ll s, e;
        Line line;
        int l, r;
    };

    const ll INF = 9223372036854775807LL;
    vector<Node> nodes;

    LiChaoTree(ll s, ll e) {
        nodes.push_back({s, e, {0, -INF}, -1, -1});
    }

    void add_line(int i, Line new_line) {
        ll s = nodes[i].s, e = nodes[i].e, m = (s + e) / 2;

        Line low, high;
        if(nodes[i].line.f(s) < new_line.f(s)) {
            low = nodes[i].line;
            high = new_line;
        } else {
            low = new_line;
            high = nodes[i].line;
        }

        // One is above the other.
        if(low.f(e) <= high.f(e)) {
            nodes[i].line = high;
        }
        // Intersect on the right.
        else if(low.f(m) < high.f(m)) {
            nodes[i].line = high;
            if(nodes[i].r == -1) {
                nodes[i].r = nodes.size();
                nodes.push_back({m + 1, e, {0, -INF}, -1, -1});
            }
            add_line(nodes[i].r, low);
        }
        // Intersect on the left.
        else {
            nodes[i].line = low;
            if(nodes[i].l == -1) {
                nodes[i].l = nodes.size();
                nodes.push_back({s, m, {0, -INF}, -1, -1});
            }
            add_line(nodes[i].l, high);
        }
    }

    ll get_max_y(int i, ll x) {
        if(i == -1) return -INF;
        ll s = nodes[i].s, e = nodes[i].e, m = (s + e) / 2;
        if (x <= m)
            return max(nodes[i].line.f(x), get_max_y(nodes[i].l, x));
        else
            return max(nodes[i].line.f(x), get_max_y(nodes[i].r, x));
    }
};
```

### Merge Sort Tree
```C++
#include <algorithm>
#include <vector>

struct Node
{
    std::vector<int> subArr;
    int left, right;
    Node* leftChild = nullptr;
    Node* rightChild = nullptr;
};

void mergeSubArray(std::vector<int> &v1, std::vector<int> &v2, std::vector<int> &dest)
{
    dest.resize(v1.size() + v2.size());
    size_t i1 = 0, i2 = 0, pos = 0;

    while (i1 < v1.size() && i2 < v2.size())
    {
        if (v1[i1] <= v2[i2])
            dest[pos++] = v1[i1++];
        else
            dest[pos++] = v2[i2++];
    }

    while (i1 < v1.size())
        dest[pos++] = v1[i1++];
    while (i2 < v2.size())
        dest[pos++] = v2[i2++];
}

Node* buildNode(int left, int right, std::vector<int>& arr)
{
    Node *current = new Node;
    current->left = left;
    current->right = right;

    if (left == right)
        current->subArr.push_back(arr[left]);
    
    else 
    {
        int mid = (left+right)/2;
        Node* leftChild = buildNode(left, mid, arr);
        Node* rightChild = buildNode(mid+1, right, arr);
        mergeSubArray(leftChild->subArr, rightChild->subArr, current->subArr);
        current->leftChild = leftChild;
        current->rightChild = rightChild;
    }

    return current;
}

int countBigger(Node* current, int threshold, int left, int right)
{
    if (current->right < left || right < current->left)
        return 0;

    if (left <= current->left && current->right <= right)
    {
        auto found = std::upper_bound(current->subArr.begin(), current->subArr.end(), threshold);
        return current->subArr.end() - found;
    }

    return countBigger(current->leftChild, threshold, left, right)
        + countBigger(current->rightChild, threshold, left, right);
}
```

### Centroid
```C++
int getSz(int here,int dad){
    sz[here]=1;
    for(auto there:adj[here]){
        if(there==dad)continue;
        sz[here]+=getSz(there,here);
    }
    return sz[here];
}
 
int get_centroid(int here,int dad,int cap){
    //cap <---- (tree size)/2
    for(auto there:adj[here]){
        if(there==dad)continue;
        if(sz[there]>cap) return get_centroid(there,here,cap);
    }
    return here;
}
int main(){
    int root=1;
    getSz(root,-1);
    int center=get_centroid(1,-1,sz[root]/2);
    return 0;
}
// 
출처 : https://smu201111192.tistory.com/1
```

## Graph

### Dinic's Algorithm
```C++
class Dinic {
    // 간선 구조체, 반대쪽에 연결된 정점과 용량 역방향 간선의 위치를 가지고 있다.
    struct Edge {
        int v, cap, rev;
        Edge(int v, int cap, int rev) : v(v), cap(cap), rev(rev) {}
    };

    const int INF = 987654321;
    int MAX_V;
    int S, E;  // source, sink
    vector<vector<Edge>> adj;
    vector<int> level, work;

    bool bfs() {
        fill(level.begin(), level.end(), -1);  // 레벨 그래프 초기화
        queue<int> qu;
        level[S] = 0;
        qu.push(S);
        while (qu.size()){
            int here = qu.front();
            qu.pop();
            for (auto i : adj[here]) {
                int there = i.v;
                int cap = i.cap;
                if (level[there] == -1 && cap > 0) {  // 레벨이 아직 정해지지 않았고 잔여용량이 0 이상
                    level[there] = level[here] + 1;  // 현재의 레벨값+1을 할당해준다.
                    qu.push(there);
                }
            }
        }
        return level[E] != -1;  // 싱크의 레벨이 할당이 안된 경우 0을 리턴
    }

    int dfs(int here, int crtcap) {
        if (here == E) return crtcap;  // 싱크일 경우 현재 흐르는 유량을 return
        for (int &i = work[here]; i < int(adj[here].size()); i++) {  // work 배열에는 다음 탐색 위치가 저장되어 있다.
            int there = adj[here][i].v;
            int cap = adj[here][i].cap;
            if (level[here] + 1 == level[there] && cap > 0) {  // 레벨 그래프가 1만큼 크고 잔여 용량이 0 이상인 간선
                int c = dfs(there, min(crtcap, cap));  // dfs로 다음 위치 탐색
                if (c > 0) {  // 싱크까지 도달하여 흐르는 차단유량이 0 이상일 경우
                    adj[here][i].cap -= c;  // 현재 용량에서 차단 유량만큼을 빼줌
                    adj[there][adj[here][i].rev].cap += c;  // 역방향 간선에 c만큼 용량을 추가해줌
                    return c;
                }
            }
        }
        return 0;
    }

public:
    Dinic(int MAX_V) : MAX_V(MAX_V) {
        adj.resize(MAX_V);
        level.resize(MAX_V);
        work.resize(MAX_V);
    }

    // 벡터의 사이즈 만큼을 넣어주어 역방향 간선의 위치를 저장한다.
    void add_edge(int s, int e, int c) {
        adj[s].emplace_back(e, c, (int)adj[e].size());
        adj[e].emplace_back(s, c, (int)adj[s].size() - 1);
    }

    int get_max_flow(int s, int e) {
        S = s, E = e;
        int res = 0;
        while (bfs()) {  // 레벨 그래프가 만들어 지는 경우에만 동작
            fill(work.begin(), work.end(), 0);
            while (1) {
                int flow = dfs(S, INF);  // 차단유량을 구하여
                if (!flow) break;
                res += flow;  // 차단 유량이 1 이상일 경우 maximum flow에 더해줌
            }
        }
        return res;
    }
};
// 주의 : 양방향 간선이 아니라 단방햔 간선일 경우 수정할 것
```

### Hopcroft-Karp
```C++
#include <vector>
#include <queue>
#include <algorithm>

using namespace std;
const int MAX = 10000;
const int INF = 1000000000;

// A[i], B[i]: 그룹의 i번 정점과 매칭된 상대편 그룹 정점 번호
int N, A[MAX], B[MAX], dist[MAX]; // dist[i]: (A그룹의) i번 정점의 레벨(?)
bool used[MAX]; // used: (A그룹의) 이 정점이 매칭에 속해 있는가?
vector<int> adj[MAX];

// 호프크로프트 카프 전용 bfs 함수: A그룹의 각 정점에 레벨을 매김
void bfs() {
    queue<int> Q;
    // 매칭에 안 속한 A그룹의 정점만 레벨 0인 채로 시작
    for (int i = 0; i < N; i++) {
        if (!used[i]) {
            dist[i] = 0;
            Q.push(i);
        } else dist[i] = INF;
    }

    // BFS를 통해 A그룹 정점에 0, 1, 2, 3, ... 의 레벨을 매김
    while (!Q.empty()) {
        int a = Q.front();
        Q.pop();
        for (int b: adj[a]) {
            if (B[b] != -1 && dist[B[b]] == INF) {
                dist[B[b]] = dist[a] + 1;
                Q.push(B[b]);
            }
        }
    }
}

// 호프크로프트 카프 전용 dfs 함수: 새 매칭을 찾으면 true
bool dfs(int a) {
    for (int b: adj[a]) {
        // 이분 매칭 코드와 상당히 유사하나, dist 배열에 대한 조건이 추가로 붙음
        if (B[b] == -1 || (dist[B[b]] == dist[a] + 1 && dfs(B[b]))) {
            // used 배열 값도 true가 됨
            used[a] = true;
            A[a] = b;
            B[b] = a;
            return true;
        }
    }
    return false;
}

int hopcroft_karp() {
    int match = 0;
    fill(A, A + MAX, -1);
    fill(B, B + MAX, -1);
    while (1) {
        // 각 정점에 레벨을 매기고 시작
        bfs();

        // 이분 매칭과 비슷하게 A그룹 정점을 순회하며 매칭 증가량 찾음
        int flow = 0;
        for (int i = 0; i < N; i++)
            if (!used[i] && dfs(i)) flow++;

        // 더 이상 증가 경로를 못 찾으면 알고리즘 종료
        if (flow == 0) break;

        // 찾았을 경우 반복
        match += flow;
    }
    return match;
}
// 전역변수 N의 크기에 주의
```

### Vertex cover

```C++
// 위에 Hopcroft-Karp에서 추가할 것
bool check_a[MAX], check_b[MAX];
vector<int> l, r;

void dfs_b(int x){
    if(check_a[x]) return;
    check_a[x] = true;
    for(auto i : adj[x]){
        check_b[i] = 1;
        dfs_b(B[i]);
    }
}

void getcover() {
    memset(check_a, 0, sizeof(check_a));
    memset(check_b, 0, sizeof(check_b));
    for (int i = 0; i < N; i++) if (A[i] == -1) dfs_b(i);
    for (int i = 0; i < N; i++) if (!check_a[i]) l.push_back(i);
    for (int i = 0; i < M; i++) if (check_b[i]) r.push_back(i);
}
// getcover()
// vector<int> l, r;
// for (int i = 0; i < N; i++) if (!check_a[i]) l.push_back(i);
// for (int i = 0; i < M; i++) if (check_b[i]) r.push_back(i);
```

### MCMF
```C++
const int MAX_V = 2002;
const int MAX_A = 1000; // 편의를 위해 나눈 것 ex) A 100개 B 100개 -> b_i + MAX_A
enum {
    SOURCE = 2001,
    SINK
};

struct edge{
    int to, cap, f, cost;
    edge *dual;
    edge(int to1, int cap1, int cost1): to(to1), cap(cap1), cost(cost1), f(0), dual(nullptr) {}
    edge(): edge(-1, 0, 0) {}
    int spare() { return cap - f; }
    void addFlow(int f1) {
        f += f1;
        dual->f -= f1;
    }
};

int N, M, ans_flow, ans_cost;
vector<edge*> adj[MAX_V];

void make_edge(int from, int to, int cap, int cost) {
    edge *e1 = new edge(to, cap, cost), *e2 = new edge(from, 0, -cost);
    e1->dual = e2;
    e2->dual = e1;
    adj[from].push_back(e1);
    adj[to].push_back(e2);
}

void mcmf() {
    while (true) {
        int prev[MAX_V], dist[MAX_V];
        bool in_queue[MAX_V] = {0};
        edge *path[MAX_V] = {0};
        fill(prev, prev+MAX_V, -1);
        fill(dist, dist+MAX_V, 1e9);

        queue<int> q;
        q.push(SOURCE);
        dist[SOURCE] = 0;
        in_queue[SOURCE] = true;

        while(!q.empty()){
            int u = q.front();
            q.pop();
            in_queue[u] = false;
            for(edge *e: adj[u]){
                int v = e->to;
                if(e->spare() > 0 && dist[v] > dist[u] + e->cost){
                    dist[v] = dist[u] + e->cost;
                    prev[v] = u;
                    path[v] = e;
                    if (!in_queue[v]) {
                        q.push(v);
                        in_queue[v] = true;
                    }
                }
            }
        }
        if(prev[SINK] == -1) break;

        int flow = 1e9;
        for(int i = SINK; i != SOURCE; i = prev[i]) {
            flow = min(flow, path[i]->spare());
        }
        for(int i = SINK; i != SOURCE; i = prev[i]) {
            ans_cost += path[i]->cost * flow;
            path[i]->addFlow(flow);
        }
        ans_flow += flow;
    }
}

int make_flow(int source, int sink) { // Edmonds-Karp
    int total = 0;
    while (true) {
        int prev[MAX_V];
        edge *path[MAX_V] = {0};
        fill(prev, prev + MAX_V, -1);
        queue<int> q;
        q.push(source);
        while(!q.empty() && prev[sink] == -1){
            int u = q.front();
            q.pop();
            for(edge *e: adj[u]){
                int v = e->to;
                if(e->spare() > 0 && prev[v] == -1){
                    q.push(v);
                    prev[v] = u;
                    path[v] = e;
                    if(v == sink) break;
                }
            }
        }
        if(prev[sink] == -1) break;
        int flow = 1e9;
        for(int i = sink; i != source; i = prev[i])
            flow = min(flow, path[i]->spare());
        for(int i = sink; i != source; i = prev[i])
            path[i]->addFlow(flow);
        total += flow;
    }
    return total;
}
// mcmf();
// cout << ans_flow << '\n';
// cout << ans_cost << '\n';
```

### 2-SAT
```C++
#include <iostream>
#include <vector>
#include <stack>
#include <algorithm>

#define MAXN 10000

using namespace std;

stack<int> st;
vector<int> adj[2*MAXN+1];
int dfsn[2*MAXN+1], dfsn_cnt = 0;
vector<vector<int> > scc;
int group[2*MAXN+1];

int dfs(int here)
{
	st.push(here);
	dfsn[here] = ++dfsn_cnt;

	int min_dfsn = dfsn[here];

	for(int there : adj[here]) {
		if(group[there] != -1) continue;

		if(!dfsn[there]) {  // not visited
			min_dfsn = min(min_dfsn, dfs(there));
		} else {  // visited
			min_dfsn = min(min_dfsn, dfsn[there]);
		}
	}

	// pop stack and grouping
	if(min_dfsn == dfsn[here]) {
		vector<int> temp;
		while(!st.empty()) {
			int node = st.top();  st.pop();
			temp.push_back(node);
			group[node] = scc.size();
			if(node == here) break;
		}
		scc.push_back(temp);
	}

	return min_dfsn;
}

int main()
{
	int n, m;
	cin >> n >> m;

	fill_n(group, 2*n+1, -1);

	for(int i=0; i<m; i++) {
		int a, b, nota, notb;

		cin >> a >> b;

		if(a < 0) {
			a = 2 * (-a);
			nota = a - 1;
		}
		else {
			a = 2 * a - 1;
			nota = a + 1;
		}

		if(b < 0) {
			b = 2 * (-b);
			notb = b - 1;
		}
		else {
			b = 2 * b - 1;
			notb = b + 1;
		}

		adj[nota].push_back(b);
		adj[notb].push_back(a);
	}

	for(int i=1; i<=2*n; i++) {
		if(!dfsn[i]) dfs(i);
	}

	// If a and nota are in the same component, print 0.
	for(int i=1; i<=2*n; i+=2) {
		if(group[i] == group[i+1]) {
			cout << 0;
			return 0;
		}
	}

	// Set values of the components.
	vector<bool> value(scc.size(), false);

	for(int i = (int)scc.size() - 1; i >= 0; i--) {
		if(value[i]) continue;

		// Set nodes in scc[i] as false and not of them as true.
		for(int node : scc[i]) {
			int not_node = (node % 2) ? (node + 1) : (node - 1);
			value[group[not_node]] = true;
		}
	}

	// Print x1, x2, ...
	cout << "1\n";
	for(int i = 1; i <= 2 * n; i+=2)
		cout << value[group[i]] << ' ';
}
```

### Bellman-Ford
```C++
#include <bits/stdc++.h>
using namespace std;
const long long INF = 1e18L;
int N, M;
bool vst[501];
long long dist[501], adj[501][501];

void bfs(int s){
    vst[s] = true;
    queue<int> q;
    q.push(s);
    while(!q.empty()){
        int u = q.front();
        q.pop();
        for(int v = 0; v < N; v++){
            if(adj[u][v] == INF || vst[v]) continue;
            vst[v] = true;
            q.push(v);
        }
    }
}

bool BellmanFord(int s){
    fill(dist, dist + N, INF);
    dist[s] = 0L;
    for (int loop = 0; loop < N - 1; loop++){
        for (int u = 0; u < N; u++){
            for (int v = 0; v < N; v++){
                if (adj[u][v] == INF) continue;
                if (dist[u] + adj[u][v] < dist[v]){
                    dist[v] = dist[u] + adj[u][v];
                }
            }
        }
    }
    for (int u = 0; u < N; u++){
        for (int v = 0; v < N; v++){
            if (vst[u] && dist[u] + adj[u][v] < dist[v]){
                return false;
            }
        }
    }
    return true;
}

int main() {
    cin >> N >> M;
    for (int i = 0; i < N; i++){
        fill(adj[i], adj[i] + N, INF);
    }
    for (int i = 0; i < M; i++){
        int u, v;
        long long w;
        cin >> u >> v >> w;
        u--; v--;
        adj[u][v] = min(adj[u][v], w);
    }
    bfs(0);
    if (BellmanFord(0)){
        for (int i = 1; i < N; i++){
            if (!vst[i] || dist[i] == INF){
                cout << -1 << '\n';
            }
            else {
                cout << dist[i] << '\n';
            }
        }
    }
    else {
        cout << -1 << '\n';
    }
}
```

### Dijkstra
```C++
const int MAX_V = 20000;
const int INF = 1e9;
typedef pair<int, int> pi;

vector<pi> adj[MAX_V];
int dist[MAX_V];

void dijkstra(int s) {
    bool found[MAX_V] = {false};
    priority_queue<pi, vector<pi>, greater<pi>> PQ;

    fill(dist, dist+MAX_V, INF);
    dist[s] = 0;
    PQ.push(pi(0, s));

    while(!PQ.empty()){
        int u = PQ.top().second;
        PQ.pop();
        if (found[u]) continue;
        found[u] = true;
        for(auto v: adj[u]){
            int next = v.first, d = v.second;
            if(dist[next] > dist[u] + d){
                dist[next] = dist[u] + d;
                PQ.push(pi(dist[next], next));
            }
        }
    }
}

int main() {
    int V, E, start;
    cin >> V >> E >> start;
    for (int i=0; i<E; i++){
        int u, v, w;
        cin >> u >> v >> w;
        u--; v--;
        adj[u].push_back(pi(v, w));
    }
    start--;
    dijkstra(start);
}
```
### Floyd-Warshall

```C++
for (int k = 0; k < N; k++)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
```

### BCC

```C++
const int MAX = 100000;
typedef pair<int, int> P;
int V, E, dcnt, dfsn[MAX];
vector<int> adj[MAX];
stack<P> S;
vector<vector<P>> bcc;
int DFS(int curr, int prev = -1){
    // 이 정점에 dfsn 부여
    // result는 DFS 트리상에서 트리, 역방향 간선으로 도달 가능한 최소의 dfsn
    int result = dfsn[curr] = ++dcnt;
    // 인접한 정점 순회
    for(int next: adj[curr]){
        // DFS 트리상에서 자신의 부모면 스킵
        if(next == prev) continue;
        // 아직 방문하지 않은 간선이면 스택에 간선 (curr, next)을 넣음
        if(dfsn[curr] > dfsn[next]) S.push(P(curr, next));
        // 역방향 간선
        if(dfsn[next] > 0) result = min(result, dfsn[next]);
        // 트리 간선
        else{
            // DFS로 이어서 탐색
            int temp = DFS(next, curr);
            result = min(result, temp);
            // next가 DFS 트리상에서 curr의 조상 노드로 갈 수 없음: 새 BCC 발견
            if(temp >= dfsn[curr]){
                // 여태 스택에 쌓여있던 간선을 빼서 BCC 구성
                // 이때, 간선 (curr, next)까지만 빼내야 함
                vector<P> currBCC;
                while(!S.empty() && S.top() != P(curr, next)){
                    currBCC.push_back(S.top());
                    S.pop();
                }
                currBCC.push_back(S.top());
                S.pop();
                bcc.push_back(currBCC);
            }
        }
    }
    // 최소 도달 가능 dfsn을 리턴
    return result;
}
 
int main(){
    // 그래프 입력받기
    scanf("%d %d", &V, &E);
    for(int i = 0; i < E; ++i){
        int u, v;
        scanf("%d %d", &u, &v);
        --u; --v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    // DFS 시도하며 BCC별로 분리
    for(int i = 0; i < V; ++i)
        if(dfsn[i] == 0) DFS(i);
    // BCC의 개수
    printf("%d\n", bcc.size());
    // 각 BCC에 속하는 간선 목록 출력
    for(auto &c: bcc){
        for(auto &p: c)
            printf("(%d, %d) ", p.first+1, p.second+1);
        puts("");
    }
}
출처 : https://m.blog.naver.com/kks227/220802704686
```


## String

### KMP
```C++
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
```

### Trie
```C++
struct Trie {
    Trie* go[26];
    Trie* fail;
    bool out;

    Trie() : out(false) {
        fill(go, go + 26, nullptr);
    }
    ~Trie() {
        for (int i = 0; i < 26; i++)
            if (go[i]) delete go[i];
    }
    void insert(const char* key) {
        if (*key == '\0') {
            out = true;
            return;
        }
        int next = *key - 'A';
        if (!go[next]) go[next] = new Trie;
        go[next]->insert(key + 1);
    }
};
```

### Aho-Corasick
```C++
#include <algorithm>
#include <vector>
#include <queue>
#include <string>

struct Trie
{
    Trie *next[26];
    Trie *fail;
    // 실제 매칭된 문자열이 필요하다면 아래 정의 사용
    // std::vector<std::string> outputs;
    // 매칭 여부만 필요하다면
    bool matched = false;

    Trie()
    {
        std::fill(next, next + 26, nullptr);
    }

    ~Trie()
    {
        for (int i = 0; i < 26; i++)
            if (next[i])
                delete next[i];
    }
    void insert(std::string &str, int start)
    {
        if ((int)str.size() <= start)
        {
            //outputs.push_back(str);
            matched = true;
            return;
        }
        int nextIdx = str[start] - 'a';
        if (!next[nextIdx])
        {
            next[nextIdx] = new Trie;
        }
        next[nextIdx]->insert(str, start + 1);
    }
};

void buildFail(Trie *root)
{
    std::queue<Trie *> q;
    root->fail = root;
    q.push(root);
    while (!q.empty())
    {
        Trie *current = q.front();
        q.pop();

        for (int i = 0; i < 26; i++)
        {
            Trie *next = current->next[i];

            if (!next)
                continue;
            if (current == root)
                next->fail = root;
            else
            {
                Trie *dest = current->fail;

                while (dest != root && !dest->next[i])
                    dest = dest->fail;

                if (dest->next[i])
                    dest = dest->next[i];
                next->fail = dest;
            }

            /*if(next->fail->outputs.size() > 0) 
                next->outputs.insert(next->outputs.end(), current->outputs.begin(), current->outputs.end());*/
            if (next->fail->matched)
                next->matched = true;

            q.push(next);
        }
    }
}

bool find(Trie *root, std::string &query)
{
    Trie *current = root;
    bool result = false;

    for (int c = 0; c < (int)query.size(); c++)
    {
        int nextIdx = query[c] - 'a';

        while (current != root && !current->next[nextIdx])
            current = current->fail;

        if (current->next[nextIdx])
            current = current->next[nextIdx];

        if (current->matched)
        {
            result = true;
            break;
        }
    }

    return result;
}
```

### Suffix Array / LCP
```C++
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
```

## Geometry

### Convex Hull
```C++
// Graham's scan - O(nlogn)
#include <stack>
#include <vector>
#include <algorithm>

using namespace std;

struct Point {
    int x, y;
};
vector<Point> p;

// 반시계 방향이면 1, 일직선이면 0, 시계 방향이면 -1.
int ccw(Point &p1, Point &p2, Point &p3) {
    long long z = (long long)p1.x * p2.y + p2.x * p3.y + p3.x * p1.y - p1.y * p2.x - p2.y * p3.x - p3.y * p1.x;
    if(z > 0)
        return 1;
    else if(z == 0)
        return 0;
    else
        return -1;
}

long long sq_dist(Point &a, Point &b) {
    return (long long)(a.x - b.x) * (a.x - b.x) + (long long)(a.y - b.y) * (a.y - b.y);
}

bool comp(Point &a, Point &b) {
    int c = ccw(p[0], a, b);
    if(c == 1)
        return true;
    else if(c == 0) {
        if(sq_dist(p[0], a) < sq_dist(p[0], b))
            return true;
        else
            return false;
    } else
        return false;
}

// Return: the number of points in the convex hull.
int convex_hull(stack<Point> &s) {
    // y 좌표가 가장 작으면서 x가 가장 작은 점 찾기.
    int n = int(p.size()), mi = 0;
    for(int i=1; i<n; i++)
        if(p[i].y < p[mi].y || (p[i].y == p[mi].y && p[i].x < p[mi].x))
            mi = i;

    swap(p[0], p[mi]);

    // 반시계 방향으로 정렬.
    sort(p.begin() + 1, p.end(), comp);

    s.push(p[0]), s.push(p[1]);
    for(int i=2; i<n; i++) {
        Point a, b, c = p[i];
        do {
            b = s.top(), s.pop();
            a = s.top();
        } while(int(s.size()) >= 2 && ccw(a, b, c) <= 0);
        if(ccw(a, b, c) == 1)
            s.push(b);
        s.push(c);
    }

    return int(s.size());
}
```

### Rotating Calipers
```C++
// O(n)
#include<algorithm>
#define x first
#define y second
#define dis(a,b) 1LL*(a.x-b.x)*(a.x-b.x)+1LL*(a.y-b.y)*(a.y-b.y)
using namespace std;
const int MXN = 2e5;
int t, n;
typedef struct pair<int, int> point;
point p[MXN], ch[MXN], ra, rb;
long long ccw(point a, point b, point c) {
    return 1LL * (b.x - a.x)*(c.y - a.y) - 1LL * (c.x - a.x)*(b.y - a.y);
}
void f() {
    scanf("%d", &n);
    for (int i = 0; i < n; i++) scanf("%d %d", &p[i].x, &p[i].y);
    swap(p[0], *min_element(p, p + n));
    sort(p + 1, p + n, [](point l, point r) {
        long long c = ccw(p[0], l, r);
        return c > 0 || !c && l<r;
    });
    int sz = 0;
    for (int i = 0; i < n; i++) {
        while (sz > 1 && ccw(ch[sz - 2], ch[sz - 1], p[i]) <= 0) sz--;
        ch[sz++] = p[i];
    }
    long long maxi = 0;
    for (int i = 0, j = 1; i < sz; i++) {
        while (ccw(ch[i], ch[(i + 1) % sz], { ch[i].x + ch[(j + 1) % sz].x - ch[j].x, ch[i].y + ch[(j + 1) % sz].y - ch[j].y }) > 0) j = (j + 1) % sz;
        if (maxi < dis(ch[i], ch[j])) {
            maxi = dis(ch[i], ch[j]);
            ra = ch[i];
            rb = ch[j];
        }
    }
    printf("%d %d %d %d\n", ra.x, ra.y, rb.x, rb.y);
}
```

## Math
### FFT
```C++
const double pi = 3.14159265358979323846264;
typedef complex<double> base;

void fft(vector<base> &a, bool invert) {
    int n = a.size();
    for (int i = 1, j = 0; i < n; i++){
        int bit = n >> 1;
        for (; j >= bit; bit >>= 1) j -= bit;
        j += bit;
        if (i < j) swap(a[i],a[j]);
    }
    for (int len = 2; len <= n; len <<= 1){
        double ang = 2 * pi / len * (invert ? -1 : 1);
        base wlen(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len){
            base w(1);
            for (int j = 0; j < len / 2; j++){
                base u = a[i + j], v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    if (invert) {
        for (int i = 0; i < n; i++) a[i] /= n;
    }
}

void multiply(const vector<long long> &a, const vector<long long> &b, vector<long long> &res) {
    vector <base> fa(a.begin(), a.end()), fb(b.begin(), b.end());
    int n = 1;
    while (n < max(a.size(), b.size())) n <<= 1;
    n <<= 1;
    fa.resize(n);
    fb.resize(n);
    fft(fa,false);
    fft(fb,false);

    for (int i = 0; i < n; i++) {
        fa[i] *= fb[i];
    }
    fft(fa,true);
    res.resize(n);
    for (int i = 0; i < n; i++) {
        res[i] = (long long)(fa[i].real() + (fa[i].real() > 0 ? 0.5 : -0.5));
    }
}

int main() {
    int n; cin >> n;
    vector<long long> x(n, 0), y(n, 0), z;
    // X = a0 * 10^0 + a1 * 10^1 + ...
    // Y = b0 * 10^0 + b1 * 10^1 + ...
    // Z = X * Y
    multiply(x, y, z);
}
```
### fermat's little theorem
```C++
long long pow(long long n, long long r, int MOD) {
    long long ret = 1;
    for (; r; r >>= 1) {
        if (r & 1) ret = ret * n % MOD;
        n = n * n % MOD;
    }
    return ret;
}
/* A * B^(p−2)  (mod p)
long long ans = fact[a];
ans = ans * pow(fact[b], MOD - 2, MOD) % MOD;
ans = ans * pow(fact[a - b], MOD - 2, MOD) % MOD;
*/
```

### Extended Euclidian Algorithm
```C++
struct INFO {
    int gcd;
    int s;
    int t;
};
vector<int> s, t, r, q;
INFO xGCD(int a, int b) {
    s = { 1,0 };
    t = { 0,1 };
    r = { a,b };
    while (1)
    {
        int r2 = r[r.size() - 2];
        int r1 = r[r.size() - 1];
        q.push_back(r2 / r1);
        r.push_back(r2 % r1);
        
        if (r[r.size() - 1] == 0)
            break;
 
        int s2 = s[s.size() - 2];
        int s1 = s[s.size() - 1];
 
        int t2 = t[t.size() - 2];
        int t1 = t[t.size() - 1];
        
        int q1 = q[q.size() - 1];
        s.push_back(s2 - s1 * q1);
        t.push_back(t2 - t1 * q1);
    }
    // return gcd, s, t
    INFO ret = { r[r.size() - 2], s[s.size() - 1], t[t.size() - 1] };
    s.clear(), t.clear(), r.clear(), q.clear();
    return ret;
}
int main()
{
    int a, b;
    scanf("%d %d", &a, &b);
    if (b > a)
        swap(a, b);
    INFO ret = xGCD(a, b);
    printf("gcd :: %d s :: %d t :: %d", ret.gcd, ret.s, ret.t);
    return 0;
}
출처: https://www.crocus.co.kr/1232 [Crocus]
```

## Dynamic Programming

### Knuth
```C++
// Knuth Optimization을 적용하기 위한 조건

/*
조건 1: dp[i][j] = min_{i<k<j} (dp[i][k] + dp[k][j]) + C[i][j]
조건 2: C[a][c] + C[b][d] <= C[a][d] + C[b][c] (a <= b <= c <= d)
조건 3: C[b][c] <= C[a][d] (a <= b <= c <= d)
*/

// 위 세 조건이 만족될 경우, O(N^2)으로 해결가능

#include <algorithm>
#include <vector>
#include <limits>

typedef long long Long;
const Long INF = 1LL<<32;
int data[1003];
Long d[1003][1003];
int p[1003][1003];

int getCost(int left, int right)
{
    // define your cost function here
}

// data의 [left, right]에 값을 채우고 아래 함수를 실행하면 d에 dp값이 채워진다.
void doKnuthOpt(int left, int right)
{
    for (int i = left; i <= right; i++)
	{
		d[i][i] = 0, p[i][i] = i;
		for (int j = i + 1; j <= right; j++)
			d[i][j] = 0, p[i][j] = i;
	}

	for (int l = 2; l <= right-left+1; l++) 
	{
		for (int i = left; i + l <= right; i++) 
		{
			int j = i + l;
			d[i][j] = INF;
			for (int k = p[i][j - 1]; k <= p[i + 1][j]; k++) 
			{
                int current = d[i][k] + d[k][j] + getCost(i, j);
				if (d[i][j] > current) 
				{
					d[i][j] = current;
					p[i][j] = k;
				}
			}
		}
	}
}
```

### Divide and Conquer
```C++
// DnQ Optimization을 적용하기 위한 조건

/*
조건 1: dp[t][i] = min_{k<i} (dp[t-1][k] + C[k][i])
*/

/*
조건 2: 아래 두 조건들 중 적어도 하나를 만족

    a)  A[t][i]를 dp[t][i]를 만족시키는 최소의 k라고 할 때 아래 부등식을 만족
        A[t][i] <= A[t][i+1]
    
    b)  비용 C가 a<=b<=c<=d인 a, b, c, d에 대하여
        사각부등식 C[a][c] + C[b][d] <= C[a][d] + C[b][c] 를 만족
*/

// 위 두 조건이 만족될 경우, O(KN log N)으로 해결가능

#include <iostream>
#include <algorithm>

typedef long long Long;

int L, G;
Long Ci[8001];
Long sum[8001];

Long dp[801][8001], properK[801][8001];

// 문제에 맞게 Cost 정의
Long Cost(Long a, Long b)
{
    return (sum[b] - sum[a - 1]) * (b - a + 1);
}

// dp[t][i] = min_{k<i} (dp[t-1][k] + C[k][i]) 꼴의 문제를 풀고자 할 때,
// 아래 함수는 dp[t][l~r]을 채운다.
void Find(int t, int l, int r, int p, int q)
{
    if (l > r)
        return;

    int mid = (l + r) >> 1;
    dp[t][mid] = -1;
    properK[t][mid] = -1;

    for (int k = p; k <= q; ++k)
    {
        Long current = dp[t - 1][k] + Cost(k+1, mid);
        if (dp[t][mid] == -1 || dp[t][mid] > current)
        {
            dp[t][mid] = current;
            properK[t][mid] = k;
        }
    }

    Find(t, l, mid - 1, p, properK[t][mid]);
    Find(t, mid + 1, r, properK[t][mid], q);
}
```

### Convex hull trick
```C++
struct Line {
    long long a, b;
    Line(long long a, long long b) : a(a), b(b) {}
};

inline bool x_on_right(Line &l0, Line &l1, long long x) {
    return l1.b - l0.b <= (l0.a - l1.a) * x;
}

inline bool l2_on_right(Line &l0, Line &l1, Line &l2) {
    return (l1.b - l0.b) * (l1.a - l2.a) < (l2.b - l1.b) * (l0.a - l1.a);
}

int main() {
	// ...
    vector<long long> dp(N + 1);
    vector<Line> st;
    st.emplace_back(y[1], dp[0]);  // a, b

    for (int i = 1; i <= N; i++) {
        Line l = st.front();
        if (st.size() >= 2) {  // 직선이 2개 이상일 때만
            int low = 0, high = int(st.size()) - 2;
            while (low <= high) {  // 이분 탐색
                int mid = (low + high) / 2;
                Line l0 = st[mid], l1 = st[mid + 1];
                if (x_on_right(l0, l1, x[i])) {  // x[i]가 l0, l1 교점 오른쪽에 있을 때
                    l = l1;
                    low = mid + 1;
                } else {
                    high = mid - 1;
                }
            }
        }

        dp[i] = l.a * x[i] + l.b;

        Line l2 = {y[i + 1], dp[i]};
        while (st.size() >= 2) {
            int e = int(st.size()) - 1;
            Line l0 = st[e - 1], l1 = st[e];
            if (l2_on_right(l0, l1, l2)) {
                break;
            }
            st.pop_back();
        }
        st.push_back(l2);
    }
    cout << dp[N];
}
```

### iknoom's CHT

```C++
#include <bits/stdc++.h>

using namespace std;
const int MAX_SIZE = 1000001;

/* 교점을 분수로 보관
struct LinearFunc {
    long long a, b, u, d;
    LinearFunc(): LinearFunc(1, 0){}
    LinearFunc(long long a1, long long b1): a(a1), b(b1), u(-1e18), d(1){}
};

bool cross_after(LinearFunc &f, LinearFunc &g) {
    g.u = f.b - g.b;
    g.d = g.a - f.a;
    if (g.d < 0) {
        g.d = -g.d;
        g.u = -g.u;
    }
    return f.u * g.d < g.u * f.d;
}
 */

struct LinearFunc {
    long long a, b;
    double s;
    LinearFunc(): LinearFunc(1, 0){}
    LinearFunc(long long a1, long long b1): a(a1), b(b1), s(-1e9){}
};

double cross(LinearFunc &f, LinearFunc &g) {
    return (g.b - f.b) / (f.a - g.a);
}

int main() {
    long long n, a[MAX_SIZE], b[MAX_SIZE];
    cin >> n;
    for (int i = 0; i < n; i++) cin >> a[i];
    for (int i = 0; i < n; i++) cin >> b[i];

    // dp[i] = b[j]a[i] + dp[j]
    long long dp[MAX_SIZE];
    vector<LinearFunc> stack;
    for(int i = 1; i < n; i++){
        // 새로운 직선을 스택에 push
        LinearFunc g(b[i-1], dp[i-1]);
        while(!stack.empty()){
            // 교점을 구하면서 교점이 뒤에 있는지 검사함
            g.s = cross(stack.back(), g);
            if(stack.back().s < g.s) break;
            stack.pop_back();
        }
        stack.push_back(g);
        long long x = a[i];
        int pos;
        if(x < stack.back().s){
            // x에 해당하는 선분을 이분 탐색으로 찾음
            int lo = 0, hi = stack.size() - 1;
            while(lo + 1 < hi){
                int mid = (lo+hi)/2;
                (x < stack[mid].s ? hi : lo) = mid;
            }
            pos = lo;
        } else {
            // 마지막 선분
            pos = stack.size() - 1;
        }
        // dp 값 계산
        dp[i] = stack[pos].a * x + stack[pos].b;
    }
    cout << dp[n-1] << '\n';
}
```

## Etc.

### Mo's Algorithm
```C++
#include <algorithm>
#include <vector>
#include <cmath>

struct Query
{
    static int sqrtN;
    int start, end, index;
    
    bool operator<(const Query& q) const
    {
        if (start / sqrtN != q.start / sqrtN)
            return start / sqrtN < q.start / sqrtN;
        else return end < q.end;
    }
};
int Query::sqrtN = 0;

std::vector<int> mosAlg(std::vector<int>& arr, std::vector<Query>& queries)
{
    // sqrt(arr의 크기)로 구간을 나누어 정렬
    std::sort(queries.begin(), queries.end());

    // 이 아래부터는 문제에 따라 다른 구현을 해야 함.
    // 이전에 쿼리한 구간에서 양쪽을 새 구간으로 맞추어서 결과를 구함.
    // 아래는 쿼리한 구간에서 존재하는 서로 다른 수의 개수를 구하는 예시 (BOJ 13547)
    int currCount = 0;
    std::vector<int> count(*std::max_element(arr.begin(), arr.end()) + 1);
    std::vector<int> answer(queries.size());
    int start = queries[0].start, end = queries[0].end;

    for (int i = start; i < end; ++i)
    {
        ++count[arr[i]];
        if (count[arr[i]] == 1)
            ++currCount;
    }
    answer[queries[0].index] = currCount;

    for (int i = 1; i < (int)queries.size(); ++i)
    {
        while (queries[i].start < start)
        {
            ++count[arr[--start]];
            if (count[arr[start]] == 1)
                ++currCount;
        }

        while (end < queries[i].end)
        {
            ++count[arr[end]];
            if (count[arr[end++]] == 1)
                ++currCount;
        }

        while (start < queries[i].start)
        {
            --count[arr[start]];
            if (count[arr[start++]] == 0)
                --currCount;
        }

        while (queries[i].end < end)
        {
            --count[arr[--end]];
            if (count[arr[end]] == 0)
                --currCount;
        }
        
        answer[queries[i].index] = currCount;
    }

    return answer;
}
```

### 히스토그램에서 가장 큰 직사각형
```C++
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
```

### 가장 가까운 두 점
```C++
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

#define MAX 800000000

using namespace std;

struct coo {
	int x, y;
};

int square(int a);
int dist2(coo a, coo b);
bool comp_x(coo a, coo b);
bool comp_y(coo a, coo b);

// [left, right)
int find_min(vector<coo>& p, int left, int right)
{
	if(left >= right - 1) return MAX;

	int left_min = find_min(p, left, (left+right)/2);
	int right_min = find_min(p, (left+right)/2, right);

	int min_square = min(left_min, right_min);
	double width = sqrt(min_square);

	double mid_x = (p[(left+right-1)/2].x + p[(left+right)/2].x) / 2.0;
	double left_x = mid_x - width, right_x = mid_x + width;

	vector<int> xs(right-left);
	for(int i = left; i < right; i++) {
		xs[i-left] = p[i].x;
	}

	// Find an index at which left_x < p[index].x
	int left_idx = upper_bound(xs.begin(), xs.end(), floor(left_x)) - xs.begin() + left;

	// Find an index at which p[index].x < right_x
	int right_idx = lower_bound(xs.begin(), xs.end(), ceil(right_x)) - xs.begin() + left;

	// [left_idx, right_idx)
	if(right_idx - left_idx <= 1) return min_square;

	vector<coo> p_in(right_idx-left_idx);
	for(int i = left_idx; i < right_idx; i++) {
		p_in[i-left_idx] = p[i];
	}
	sort(p_in.begin(), p_in.end(), comp_y);

	int center_min = MAX, bot = 0;
	for(int i = 1; i < right_idx-left_idx; i++) {
		while(square(p_in[i].y-p_in[bot].y) >= min_square && bot < i) bot++;
		for(int j = bot; j < i; j++) {
			center_min = min(center_min, dist2(p_in[i], p_in[j]));
		}
	}

	return min(min_square, center_min);
}

int main()
{
	int n;
	cin >> n;

	vector<coo> p(n);
	for(int i = 0; i < n; i++)
		cin >> p[i].x >> p[i].y;

	sort(p.begin(), p.end(), comp_x);

	cout << find_min(p, 0, n);
}

int square(int a)
{
	return a * a;
}

int dist2(coo a, coo b)
{
	return square(a.x - b.x) + square(a.y - b.y);
}

bool comp_x(coo a, coo b)
{
	return a.x < b.x;
}

bool comp_y(coo a, coo b)
{
	return a.y < b.y;
}
```

### 좌표 압축
```C++
vector<int> s;
for(int i=1; i<=n; i++) s.push_back(Q[i]);
sort(s.begin(), s.end());
s.erase(unique(s.begin(), s.end()), s.end());
for(int i=1; i<=n; i++)
    Q[i] = lower_bound(s.begin(), s.end(), Q[i]) - s.begin();
```

### iknoom's segtree
```C++
const int SIZE = 2097152;
struct SegTree{
    int size, start;
    long long arr[SIZE];

    SegTree(int n): size(n){
        start = 1;
        while (start < size) start *= 2;
        memset(arr, 0, sizeof(arr));
    }
    void prepare(){
        for(int i = start - 1; i; i--) arr[i] = arr[i * 2] + arr[i * 2 + 1];
    }
    void update(int here, long long val){
        here += start;
        arr[here] = val;
        while (here){
            here /= 2;
            arr[here] = arr[here * 2] + arr[here * 2 + 1];
        }
    }
    long long sum(int l, int r){
        l += start;
        r += start;
        long long ret = 0;
        while (l <= r){
            if (l % 2 == 1) ret += arr[l++];
            if (r % 2 == 0) ret += arr[r--];
            l /= 2; r /= 2;
        }
        return ret;
    }
    int search(int k) {
        int pos = 1;
        while(pos < start){
            if(k <= arr[pos << 1]) pos <<= 1;
            else k-=arr[pos << 1], pos = pos << 1 | 1;
        }
        return pos-start;
    }
    void update(int lo, int hi, int val, int node, int x, int y) {
        if (y < lo || hi < x)
            return;
        if (lo <= x&&y <= hi)
            cnt[node] += val;
        else {
            int mid = (x + y) >> 1;
            update(lo, hi, val, node * 2, x, mid);
            update(lo, hi, val, node * 2 + 1, mid + 1, y);
        }
        if (cnt[node])seg[node] = y - x + 1;
        else {
            if (x == y)seg[node] = 0;
            else seg[node] = seg[node * 2] + seg[node * 2 + 1];
        }
    }
};
// 
const int INF = 1e9;

struct Node{
    int s, m, l, r;
    Node() : s(0), m(-INF), l(-INF), r(-INF) { }
    Node operator+(Node &right) {
        Node ret;
        ret.s = s + right.s;
        ret.l = max(l, s + right.l);
        ret.r = max(right.r, r + right.s);
        ret.m = max(r + right.l, max(m, right.m));
        return ret;
    }
};

struct SegmentTree{
    vector<Node> data;
    int n;
    SegmentTree(int size) {
        int p = 1;
        while (p < size) p *= 2;
        data.resize(p * 2);
        n = p;
    }

    void set(int i, int v) {
        i += n - 1;
        data[i].s = v;
        data[i].l = v;
        data[i].r = v;
        data[i].m = v;
    }

    void build() {
        for (int i = n - 2; i >= 0; i--) {
            data[i] = data[i * 2 + 1] + data[i * 2 + 2];
        }
    }

    void update(int i, int v) {
        set(i, v);
        i += n - 1;
        while (i > 0) {
            i = (i - 1) / 2;
            data[i] = data[i * 2 + 1] + data[i * 2 + 2];
        }
    }

    int query(int l, int r){
        l += n;
        r += n + 1;
        Node ret_l = Node();
        Node ret_r = Node();
        while (l < r) {
            if (r & 1) {
                r--;
                ret_r = data[r - 1] + ret_r;
            }
            if (l & 1) {
                ret_l = ret_l + data[l - 1];
                l++;
            }
            l >>= 1;
            r >>= 1;
        }
        return (ret_l + ret_r).m;
    }
};
```

