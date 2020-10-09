# 2020 ACM-ICPC ManManChiAnunTeam Team Note

## Tree

### LCA
```C++
#include <iostream>
#include <vector>

#define MAXN 30001
#define MAX_PSIZE 15

using namespace std;

int n, p[MAXN][MAX_PSIZE], d[MAXN];
vector<int> adj[MAXN];
bool v[MAXN];

void dfs(int i, int depth) {
    v[i] = true, d[i] = depth;
    for (int j : adj[i])
        if (!v[j]) {
            p[j][0] = i;
            dfs(j, depth + 1);
        }
}

void compute_p() {
    for (int j = 1; j < MAX_PSIZE; j++)
        for (int i = 1; i <= n; i++)
            p[i][j] = p[p[i][j - 1]][j - 1];
}

int find_lca(int a, int b) {
    // Make a have a higher depth
    if (d[a] < d[b]) swap(a, b);

    // Elevate a to the depth of b
    int depth_diff = d[a] - d[b];
    for (int j = MAX_PSIZE - 1; j >= 0; j--)
        if (depth_diff & (1 << j))
            a = p[a][j];

    if (a == b) return a;

    for (int j = MAX_PSIZE - 1; j >= 0; j--)
        if (p[a][j] != p[b][j])
            a = p[a][j], b = p[b][j];

    if (p[a][0] == p[b][0] && p[a][0]) return p[a][0];
    else return -1;
}

int main() {
    cin >> n;
    for (int i = 0; i < n - 1; i++) {
        int x, y;
        cin >> x >> y;
        adj[x].push_back(y), adj[y].push_back(x);
    }

    for (int i = 1; i <= n; i++)
        if (!v[i]) dfs(i, 0);

    compute_p();

    int m, x, y, total = 0;
    cin >> m;
    cin >> x;
    for (int i = 1; i < m; i++) {
        cin >> y;
        int lca = find_lca(x, y);
        if (lca != -1) total += d[x] + d[y] - 2 * d[lca];
        x = y;
    }

    cout << total;
}
```

### Segment Tree
```C++
class SegTree {
    struct Node {
        int b, e;
        long long s;
        Node *l, *r;
        Node(int b, int e) : b(b), e(e), s(0), l(nullptr), r(nullptr) {}
    };

    Node *root;

    void update_(Node *now, int i, int value) {
        if (now->b == now->e) {
            now->s = value;
            return;
        }
        int mid = (now->b + now->e) / 2;
        if (i <= mid) {
            if (!now->l) now->l = new Node(now->b, mid);
            update_(now->l, i, value);
        } else {
            if (!now->r) now->r = new Node(mid + 1, now->e);
            update_(now->r, i, value);
        }
        now->s = (now->l ? now->l->s : 0) + (now->r ? now->r->s : 0);
    }

    long long get_sum_(Node *now, int b, int e) {
        if (!now || e < now->b || now->e < b)
            return 0;  // out of range
        else if (b <= now->b && now->e <= e)
            return now->s;  // included
        else
            return get_sum_(now->l, b, e) + get_sum_(now->r, b, e);
    }

    void delete_nodes(Node *now) {
        if (!now) return;
        delete_nodes(now->l);
        delete_nodes(now->r);
        delete now;
    }

public:
    SegTree(int N) { root = new Node(1, N); }
    void update(int i, int value) { update_(root, i, value); }
    long long get_sum(int b, int e) { return get_sum_(root, b, e); }
    ~SegTree() { delete_nodes(root); }
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
    cin.tie(nullptr); cout.tie(nullptr); ios::sync_with_stdio(false);
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

## String

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

## Math

## Dynamic Programming

## Etc.

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
