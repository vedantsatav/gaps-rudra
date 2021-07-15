#include <iostream>
#include <bits/stdc++.h>

using namespace std;

void change(vector<int> (a)[8], int i)
{
    a[1].push_back(123);
}

int main()
{
    vector<int> a[8];

    // std::array<std::vector<int>, 4> a;

    for(int i = 0; i < 8; i++)
    {
        a[i].push_back(0);
        a[i].push_back(1);
    }

    for(int i = 0; i < 8; i++)
    {
        change(a,i);
    }

    for(int i = 0; i < 8; i++)
    {
        vector<int> b = a[i];
        for(int j = 0; j < a[i].size(); j++)
        {
            cout << b[j] << " ";
        }
        cout << endl;
    }

    return 0;
}