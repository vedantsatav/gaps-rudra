// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <algorithm>
#include <iostream>
#include <vector>
#include <cmath>
#include "benchmark.h"
#include "builder1.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"
#include <unordered_map>
#include "System.hpp"
#include <fstream>
/*
GAP Benchmark Suite
Kernel: PageRank (PR)
Author: Scott Beamer

Will return pagerank scores for all vertices once total change < epsilon

This PR implementation uses the traditional iterative approach. This is done
to ease comparisons to other implementations (often use same algorithm), but
it is not necesarily the fastest way to implement it. It does perform the
updates in the pull direction to remove the need for atomics.
*/

using namespace std;

typedef float ScoreT;
const float kDamp = 0.85;
uint64_t return_cachelines(const Graph &g)
{

  struct Data
  {
    uint64_t val;
    uint64_t num;
  };
  ofstream file;
  file.open("degree_cache.txt");
  unordered_map<int, Data> myMap;

  uint64_t number_of_cache_lines = 0;
  for (NodeID u = 0; u < g.num_nodes(); u++)
  {

    // unordered_map<NodeID,int> umap;
    // unordered_map<NodeID,int> umap;

    unordered_map<NodeID, bool> umap;
    for (NodeID v : g.out_neigh(u))
    {
      NodeID set_number = ceil(v / 8);
      umap[set_number] = true;
    }
    myMap[g.out_degree(u)].val += umap.size();
    myMap[g.out_degree(u)].num += 1;

    // myMap[u].num_cache_line=umap.size();

    number_of_cache_lines += umap.size();
  }

  for (auto i = myMap.begin(); i != myMap.end(); i++)
    file << "Degree - " << i->first << ",Cache Line - " << i->second.val << " Number- " << i->second.num << ",Average - " << i->second.val / i->second.num << endl;

  //  for (auto x : myMap)
  // {
  //   file <<"   Degree= "<<x.val<<", Number= "<<x.num<<"Average"<<   x.val/x.num<<endl;
  // }

  file.close();
  return number_of_cache_lines;
}

pvector<ScoreT> PageRankPull(const Graph &g, int max_iters,
                             double epsilon = 0)
{

  // System::profile("queries", [&]() {
  const ScoreT init_score = 1.0f / g.num_nodes();
  const ScoreT base_score = (1.0f - kDamp) / g.num_nodes();
  pvector<ScoreT> scores(g.num_nodes(), init_score);
  pvector<ScoreT> outgoing_contrib(g.num_nodes());
  for (int iter = 0; iter < max_iters; iter++)
  {
    double error = 0;
#pragma omp parallel for
    for (NodeID n = 0; n < g.num_nodes(); n++)
      outgoing_contrib[n] = scores[n] / g.out_degree(n);
#pragma omp parallel for reduction(+ \
                                   : error) schedule(dynamic, 64)
    for (NodeID u = 0; u < g.num_nodes(); u++)
    {
      ScoreT incoming_total = 0;
      for (NodeID v : g.in_neigh(u))
        incoming_total += outgoing_contrib[v];
      ScoreT old_score = scores[u];
      scores[u] = base_score + kDamp * incoming_total;
      error += fabs(scores[u] - old_score);
    }
    printf(" %2d    %lf\n", iter, error);
    if (error < epsilon)
      break;
  }

  // for(NodeID u = 0; u < g.num_nodes(); u++)
  // {
  //   cout << "scores[" << u << "]: " << scores[u] << endl; 
  // }
  return scores;

  // });
}

void PrintTopScores(const Graph &g, const pvector<ScoreT> &scores)
{
  vector<pair<NodeID, ScoreT>> score_pairs(g.num_nodes());
  for (NodeID n = 0; n < g.num_nodes(); n++)
  {
    score_pairs[n] = make_pair(n, scores[n]);
  }
  int k = 5;
  vector<pair<ScoreT, NodeID>> top_k = TopK(score_pairs, k);
  k = min(k, static_cast<int>(top_k.size()));
  for (auto kvp : top_k)
    cout << kvp.second << ":" << kvp.first << endl;
}

// Verifies by asserting a single serial iteration in push direction has
//   error < target_error
bool PRVerifier(const Graph &g, const pvector<ScoreT> &scores,
                double target_error)
{
  const ScoreT base_score = (1.0f - kDamp) / g.num_nodes();
  pvector<ScoreT> incomming_sums(g.num_nodes(), 0);
  double error = 0;
  for (NodeID u : g.vertices())
  {
    ScoreT outgoing_contrib = scores[u] / g.out_degree(u);
    for (NodeID v : g.out_neigh(u))
      incomming_sums[v] += outgoing_contrib;
  }
  for (NodeID n : g.vertices())
  {
    error += fabs(base_score + kDamp * incomming_sums[n] - scores[n]);
    incomming_sums[n] = 0;
  }
  PrintTime("Total Error", error);
  return error < target_error;
}

int main(int argc, char *argv[])
{
  CLPageRank cli(argc, argv, "pagerank", 1e-4, 20, 0);
  if (!cli.ParseArgs())
    return -1;
  Builder b(cli);
  int flag = cli.flag_reordering();
  Graph g, g1;
  g1 = b.MakeGraph();
  pvector<NodeID> new_ids(g1.num_nodes(), -1);
  // double reordering_time = 0;

  Timer t;

  if (flag == 0)
  {
    g = b.MakeGraph();

  }
  else if (flag == 1)
  {
     // g1 = b.MakeGraph();
    t.Start();
    // g = Builder::generateNeighbourMapping(g1, true); //second arg = true to useOutDeg
    g = Builder::generateNumberNodeNeighbourMapping(g1, true, new_ids);
    t.Stop();
  }
  // else if (flag == 2)
  // {
  //   Graph g1 = b.MakeGraph();
  //   t.Start();
  //   g = Builder::generateParallelNeighbourMapping(g1, true); //second arg = true to useOutDeg
  //   t.Stop();
  // }
  else if (flag == 2)
  {

     // g1 = b.MakeGraph();
    t.Start();
    g = Builder::generateDBGMapping(g1, true, new_ids); //second arg = true to useOutDeg
    t.Stop();
  }
  else if (flag == 3)

  {
//for all sideways numbering
 // g1 = b.MakeGraph();
    t.Start();
    // g = Builder::generateNeighbourMapping(g1, true); //second arg = true to useOutDeg
    g = Builder::generateNumberNodeNeighbourMapping2(g1, true, new_ids);
    t.Stop();





  }

  else if(flag == 4)
  {
    // Graph g1 = b.MakeGraph();
    t.Start();
    g = Builder::generateDBGMappingUndirected(g1, true, new_ids); //second arg = true to useOutDeg
    t.Stop();
  }
  else if(flag == 5)
  {
    // Graph g1 = b.MakeGraph();
    t.Start();
    g = Builder::undirectedgenerateNumberNodeNeighbourMapping(g1, true, new_ids); //second arg = true to useOutDeg
    t.Stop();
  }
  else if(flag == 6)
  {
    // Graph g1 = b.MakeGraph();
    t.Start();
    g = Builder::generateConnectedComponentNeighbourMapping(g1, true, new_ids); //second arg = true to useOutDeg
    t.Stop();
  }
  

  else if (flag == 7)
  {
    t.Start();
    g = Builder::generateSOrdering(g1, true, new_ids); //second arg = true to useOutDeg
    t.Stop();
  }

  else if (flag == 8)
  {
    
    t.Start();
    g = Builder::generateNOrdering(g1, true, new_ids);
    t.Stop();
  }

  else if (flag == 9)
  {
    t.Start();
    g = Builder::MyReordering(g1, true, new_ids); //second arg = true to useOutDeg
    t.Stop();
  }


  else if (flag == 10)
  {
    t.Start();
    //int radius = cli.return_radius();
    //cout<<"Passing radius= "<<radius<<endl;
    g = Builder::MyReordering_alt(g1, true, new_ids); //second arg = true to useOutDeg
    t.Stop();
  }
  // uint64_t num_cacheLines = return_cachelines(g);

  // cout << "Number of Cache Lines used" << num_cacheLines;
  auto PRBound = [&cli](const Graph &g) {
    return PageRankPull(g, cli.max_iters(), cli.tolerance());
  };
  auto VerifierBound = [&cli](const Graph &g, const pvector<ScoreT> &scores) {
    return PRVerifier(g, scores, cli.tolerance());
  };
  cout << "Reorder time from PR application: " << t.Seconds() << endl;
  BenchmarkKernel(cli, g, PRBound, PrintTopScores, VerifierBound, t.Seconds());
  

  return 0;
}
