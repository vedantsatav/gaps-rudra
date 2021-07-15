#include <iostream>
#include <omp.h>
#include <bits/stdc++.h>

using namespace std;

int main()
{
    omp_set_num_threads(4);
    int max_elements[omp_get_num_threads()] = {-255};
    int max_indices[omp_get_num_threads()] = {-1};

    
    for(int i = 0; i < omp_get_num_threads(); i++)
    {
        max_elements[i] = -255;
        max_indices[i] = -1;
    }

    cout << "OpenMP num threads = " << omp_get_num_threads() << endl;


    int max_ele=-255;
    int max_in=-1;

    int arr1[] = {-1, 2, 0, 45, 67, -9};
    #pragma omp parallel for 
    for(int i = 0; i < 6; i++)
    {
        if(max_elements[omp_get_thread_num()] < arr1[i])
        {
            max_elements[omp_get_thread_num()] = arr1[i];
            max_indices[omp_get_thread_num()] = i;
        }
    }

    for(int i = 0; i < omp_get_num_threads(); i++)
    {
        cout << max_elements[i] << "\t" << max_indices[i] << endl;
    }

    for(int i = 0; i < omp_get_num_threads(); i++)
    {
        if(max_elements[i] > max_ele)
        {
            max_ele = max_elements[i];
            max_in = max_indices[i];
        }
    }

    cout << "Max element is " << max_ele << " Max index " << max_in << endl;
    return 0;
}