// #include <iostream>
#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

static void pushToBucket(vector<uint32_t>  (local_buckets)[8])
{
//   printLocalBuckets(local_buckets);
  int num_buckets = 8;
  uint16_t av = 5;

  int count = 32;

  uint32_t bucket_threshold[] = {av / 2, av, av * 2, av * 4, av * 8, av * 16, av * 32, av * 64, av * 128, av * 256, av * 512, static_cast<uint32_t>(-1)};
  bucket_threshold[num_buckets - 1] = static_cast<uint32_t>(-1);

  // cout << "Pushing node_num: " << node_num << endl;
  
  for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
  {
    if (count <= bucket_threshold[j])
    {
    //   cout << "node_num: " << node_num << endl;
      cout << "size of local buckets " << local_buckets[j].size() << endl;
      local_buckets[j].push_back(0);
      break;
    }
  }
}

int main()
{
    srand(5);

    int nos[] = {12, 4, 5, 6, 50, 67};

    vector<uint32_t> local_buckets[8];

    uint64_t start_node = 0;
    int64_t max_degree = 0;
#pragma omp parallel
    {
    uint64_t *starting_nodes;
    starting_nodes = new uint64_t[omp_get_max_threads()];

    uint64_t *rand_nos;
    rand_nos = new uint64_t[6];
    
    int64_t *max_degrees;
    max_degrees = new int64_t[omp_get_max_threads()];

    int *nums;
    nums = new int[omp_get_max_threads()];

    #pragma omp  for
    for(uint16_t i = 0; i < omp_get_max_threads(); i++)
    {
      starting_nodes[i] =0;
      max_degrees[i] = 0;
    }


    
    cout << "omp_get_max_threads(): " << omp_get_max_threads() << endl;

    int64_t num;
    // #pragma omp parallel for shared(starting_nodes, max_degrees, rand_nos, index) private(num)
    #pragma omp  for
      for (uint64_t i = 0; i < 6; i++)
      {
        // num = rand();
        // rand_nos[i] = nos[i];
        // index = i;
        // cout << "Generated random no: " << num << endl;
        if (nos[i] > max_degrees[omp_get_thread_num()])
        {
          max_degrees[omp_get_thread_num()] = num;
          starting_nodes[omp_get_thread_num()] = i;

        }
        nums[i] = omp_get_thread_num();        
      }

for(int i = 0; i < omp_get_num_threads(); i++)
    {
        cout << "i " << i << " nums " << nums[i] << endl;
    }
      for(int i = 0; i < omp_get_num_threads(); i++)
    {
      if(max_degree < max_degrees[i])
      {
        max_degree = max_degrees[i];
        start_node = starting_nodes[i];
      }
    }
    }

    // for(int i  = 0; i < 6; i++)
    // {
    //     cout << "Generated random no " << rand_nos[i] << endl;
    // }
    
    

    // for(int i = 0; i < omp_get_num_threads(); i++)
    // {
    //     cout << "i " << i << " nums " << nums[i] << endl;
    // }
    cout << "starting node : " << start_node << endl;
    cout << "max_degree : " << max_degree << endl;
    return 0;
}