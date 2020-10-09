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

## String

## Geometry

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