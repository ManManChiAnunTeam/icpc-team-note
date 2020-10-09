# 2020 ACM-ICPC ManManChiAnunTeam Team Note

## Tree

### LCA
```C++
int lca() {

}
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
    ios::sync_with_stdio(false);
    cin.tie(0);

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
	for(int i = 1; i <= 2 * n; i+=2) {
		cout << value[group[i]] << ' ';
	}

	return 0;
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
	ios::sync_with_stdio(false);
	cin.tie(0);

	int n;
	cin >> n;

	vector<coo> p(n);
	for(int i = 0; i < n; i++)
		cin >> p[i].x >> p[i].y;

	sort(p.begin(), p.end(), comp_x);

	cout << find_min(p, 0, n);

	return 0;
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