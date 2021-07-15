// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef BUILDER1_H_
#define BUILDER1_H_

#include <algorithm>
#include <cinttypes>
#include <fstream>
#include <functional>
#include <type_traits>
#include <utility>
#include <assert.h>
#include <omp.h>
#include <iostream>
#include <vector>


#include "benchmark.h"
#include "bitmap.h"
#include "pvector.h"

#include "command_line.h"
#include "generator.h"
#include "graph.h"
#include "platform_atomics.h"
#include "pvector.h"
#include "reader.h"
#include "timer.h"
#include "util.h"
#include "sliding_queue.h"
#include "bitmap.h"

#define NO_BUCKETS 8
#define NO_THREADS_BFS 8

#include <list>
using namespace std;

/*
GAP Benchmark Suite
Class:  BuilderBase
Author: Scott Beamer

Given arguements from the command line (cli), returns a built graph
 - MakeGraph() will parse cli and obtain edgelist and call
   MakeGraphFromEL(edgelist) to perform actual graph construction
 - edgelist can be from file (reader) or synthetically generated (generator)
 - Common case: BuilderBase typedef'd (w/ params) to be Builder (benchmark.h)
*/

template <typename NodeID_, typename DestID_ = NodeID_,
          typename WeightT_ = NodeID_, bool invert = true>
class BuilderBase
{
  typedef EdgePair<NodeID_, DestID_> Edge;
  typedef pvector<Edge> EdgeList;

  const CLBase &cli_;
  bool symmetrize_;
  bool needs_weights_;
  bool in_place_ = false;
  int64_t num_nodes_ = -1;

public:
  explicit BuilderBase(const CLBase &cli) : cli_(cli)
  {
    symmetrize_ = cli_.symmetrize();
    needs_weights_ = !std::is_same<NodeID_, DestID_>::value;
    in_place_ = cli_.in_place();
    if (in_place_ && needs_weights_)
    {
      std::cout << "In-place building (-m) does not support weighted graphs"
                << std::endl;
      exit(-30);
    }
  }

  DestID_ GetSource(EdgePair<NodeID_, NodeID_> e)
  {
    return e.u;
  }

  DestID_ GetSource(EdgePair<NodeID_, NodeWeight<NodeID_, WeightT_>> e)
  {
    return NodeWeight<NodeID_, WeightT_>(e.u, e.v.w);
  }

  NodeID_ FindMaxNodeID(const EdgeList &el)
  {
    NodeID_ max_seen = 0;
#pragma omp parallel for reduction(max \
                                   : max_seen)
    for (auto it = el.begin(); it < el.end(); it++)
    {
      Edge e = *it;
      max_seen = std::max(max_seen, e.u);
      max_seen = std::max(max_seen, (NodeID_)e.v);
    }
    return max_seen;
  }

  pvector<NodeID_> CountDegrees(const EdgeList &el, bool transpose)
  {
    pvector<NodeID_> degrees(num_nodes_, 0);
#pragma omp parallel for
    for (auto it = el.begin(); it < el.end(); it++)
    {
      Edge e = *it;
      if (symmetrize_ || (!symmetrize_ && !transpose))
        fetch_and_add(degrees[e.u], 1);
      if ((symmetrize_ && !in_place_) || (!symmetrize_ && transpose))
        fetch_and_add(degrees[(NodeID_)e.v], 1);
    }
    return degrees;
  }

  static pvector<SGOffset> PrefixSum(const pvector<NodeID_> &degrees)
  {
    pvector<SGOffset> sums(degrees.size() + 1);
    SGOffset total = 0;
    for (size_t n = 0; n < degrees.size(); n++)
    {
      sums[n] = total;
      total += degrees[n];
    }
    sums[degrees.size()] = total;
    return sums;
  }

  static pvector<SGOffset> ParallelPrefixSum(const pvector<NodeID_> &degrees)
  {
    const size_t block_size = 1 << 20;
    const size_t num_blocks = (degrees.size() + block_size - 1) / block_size;
    pvector<SGOffset> local_sums(num_blocks);
#pragma omp parallel for
    for (size_t block = 0; block < num_blocks; block++)
    {
      SGOffset lsum = 0;
      size_t block_end = std::min((block + 1) * block_size, degrees.size());
      for (size_t i = block * block_size; i < block_end; i++)
        lsum += degrees[i];
      local_sums[block] = lsum;
    }
    pvector<SGOffset> bulk_prefix(num_blocks + 1);
    SGOffset total = 0;
    for (size_t block = 0; block < num_blocks; block++)
    {
      bulk_prefix[block] = total;
      total += local_sums[block];
    }
    bulk_prefix[num_blocks] = total;
    pvector<SGOffset> prefix(degrees.size() + 1);
#pragma omp parallel for
    for (size_t block = 0; block < num_blocks; block++)
    {
      SGOffset local_total = bulk_prefix[block];
      size_t block_end = std::min((block + 1) * block_size, degrees.size());
      for (size_t i = block * block_size; i < block_end; i++)
      {
        prefix[i] = local_total;
        local_total += degrees[i];
      }
    }
    prefix[degrees.size()] = bulk_prefix[num_blocks];
    return prefix;
  }

  // Removes self-loops and redundant edges
  // Side effect: neighbor IDs will be sorted
  void SquishCSR(const CSRGraph<NodeID_, DestID_, invert> &g, bool transpose,
                 DestID_ ***sq_index, DestID_ **sq_neighs)
  {
    pvector<NodeID_> diffs(g.num_nodes());
    DestID_ *n_start, *n_end;
#pragma omp parallel for private(n_start, n_end)
    for (NodeID_ n = 0; n < g.num_nodes(); n++)
    {
      if (transpose)
      {
        n_start = g.in_neigh(n).begin();
        n_end = g.in_neigh(n).end();
      }
      else
      {
        n_start = g.out_neigh(n).begin();
        n_end = g.out_neigh(n).end();
      }
      std::sort(n_start, n_end);
      DestID_ *new_end = std::unique(n_start, n_end);
      new_end = std::remove(n_start, new_end, n);
      diffs[n] = new_end - n_start;
    }
    pvector<SGOffset> sq_offsets = ParallelPrefixSum(diffs);
    *sq_neighs = new DestID_[sq_offsets[g.num_nodes()]];
    *sq_index = CSRGraph<NodeID_, DestID_>::GenIndex(sq_offsets, *sq_neighs);
#pragma omp parallel for private(n_start)
    for (NodeID_ n = 0; n < g.num_nodes(); n++)
    {
      if (transpose)
        n_start = g.in_neigh(n).begin();
      else
        n_start = g.out_neigh(n).begin();
      std::copy(n_start, n_start + diffs[n], (*sq_index)[n]);
    }
  }

  CSRGraph<NodeID_, DestID_, invert> SquishGraph(
      const CSRGraph<NodeID_, DestID_, invert> &g)
  {
    DestID_ **out_index, *out_neighs, **in_index, *in_neighs;
    SquishCSR(g, false, &out_index, &out_neighs);
    if (g.directed())
    {
      if (invert)
        SquishCSR(g, true, &in_index, &in_neighs);
      return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), out_index,
                                                out_neighs, in_index,
                                                in_neighs);
    }
    else
    {
      return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), out_index,
                                                out_neighs);
    }
  }

  /*
  In-Place Graph Building Steps
    - sort edges and squish (remove self loops and redundant edges)
    - overwrite EdgeList's memory with outgoing neighbors
    - if graph not being symmetrized
      - finalize structures and make incoming structures if requested
    - if being symmetrized
      - search for needed inverses, make room for them, add them in place
  */
  void MakeCSRInPlace(EdgeList &el, DestID_ ***index, DestID_ **neighs,
                      DestID_ ***inv_index, DestID_ **inv_neighs)
  {
    // preprocess EdgeList - sort & squish in place
    std::sort(el.begin(), el.end());
    auto new_end = std::unique(el.begin(), el.end());
    el.resize(new_end - el.begin());
    auto self_loop = [](Edge e) { return e.u == e.v; };
    new_end = std::remove_if(el.begin(), el.end(), self_loop);
    el.resize(new_end - el.begin());
    // analyze EdgeList and repurpose it for outgoing edges
    pvector<NodeID_> degrees = CountDegrees(el, false);
    pvector<SGOffset> offsets = ParallelPrefixSum(degrees);
    pvector<NodeID_> indegrees = CountDegrees(el, true);
    *neighs = reinterpret_cast<DestID_ *>(el.data());
    for (Edge e : el)
      (*neighs)[offsets[e.u]++] = e.v;
    size_t num_edges = el.size();
    el.leak();
    // revert offsets by shifting them down
    for (NodeID_ n = num_nodes_; n >= 0; n--)
      offsets[n] = n != 0 ? offsets[n - 1] : 0;
    if (!symmetrize_)
    { // not going to symmetrize so no need to add edges
      size_t new_size = num_edges * sizeof(DestID_);
      *neighs = static_cast<DestID_ *>(std::realloc(*neighs, new_size));
      *index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, *neighs);
      if (invert)
      { // create inv_neighs & inv_index for incoming edges
        pvector<SGOffset> inoffsets = ParallelPrefixSum(indegrees);
        *inv_neighs = new DestID_[inoffsets[num_nodes_]];
        *inv_index = CSRGraph<NodeID_, DestID_>::GenIndex(inoffsets,
                                                          *inv_neighs);
        for (NodeID_ u = 0; u < num_nodes_; u++)
        {
          for (DestID_ *it = (*index)[u]; it < (*index)[u + 1]; it++)
          {
            NodeID_ v = static_cast<NodeID_>(*it);
            (*inv_neighs)[inoffsets[v]] = u;
            inoffsets[v]++;
          }
        }
      }
    }
    else
    { // symmetrize graph by adding missing inverse edges
      // Step 1 - count number of needed inverses
      pvector<NodeID_> invs_needed(num_nodes_, 0);
      for (NodeID_ u = 0; u < num_nodes_; u++)
      {
        for (SGOffset i = offsets[u]; i < offsets[u + 1]; i++)
        {
          DestID_ v = (*neighs)[i];
          bool inv_found = std::binary_search(*neighs + offsets[v],
                                              *neighs + offsets[v + 1],
                                              static_cast<DestID_>(u));
          if (!inv_found)
            invs_needed[v]++;
        }
      }
      // increase offsets to account for missing inverses, realloc neighs
      SGOffset total_missing_inv = 0;
      for (NodeID_ n = 0; n <= num_nodes_; n++)
      {
        offsets[n] += total_missing_inv;
        total_missing_inv += invs_needed[n];
      }
      size_t newsize = (offsets[num_nodes_] * sizeof(DestID_));
      *neighs = static_cast<DestID_ *>(std::realloc(*neighs, newsize));
      if (*neighs == nullptr)
      {
        std::cout << "Call to realloc() failed" << std::endl;
        exit(-33);
      }
      // Step 2 - spread out existing neighs to make room for inverses
      //   copies backwards (overwrites) and inserts free space at starts
      SGOffset tail_index = offsets[num_nodes_] - 1;
      for (NodeID_ n = num_nodes_ - 1; n >= 0; n--)
      {
        SGOffset new_start = offsets[n] + invs_needed[n];
        for (SGOffset i = offsets[n + 1] - 1; i >= new_start; i--)
        {
          (*neighs)[tail_index] = (*neighs)[i - total_missing_inv];
          tail_index--;
        }
        total_missing_inv -= invs_needed[n];
        tail_index -= invs_needed[n];
      }
      // Step 3 - add missing inverse edges into free spaces from Step 2
      for (NodeID_ u = 0; u < num_nodes_; u++)
      {
        for (SGOffset i = offsets[u] + invs_needed[u]; i < offsets[u + 1]; i++)
        {
          DestID_ v = (*neighs)[i];
          bool inv_found = std::binary_search(
              *neighs + offsets[v] + invs_needed[v],
              *neighs + offsets[v + 1],
              static_cast<DestID_>(u));
          if (!inv_found)
          {
            (*neighs)[offsets[v] + invs_needed[v] - 1] = static_cast<DestID_>(u);
            invs_needed[v]--;
          }
        }
      }
      for (NodeID_ n = 0; n < num_nodes_; n++)
        std::sort(*neighs + offsets[n], *neighs + offsets[n + 1]);
      *index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, *neighs);
    }
  }

  /*
  Graph Bulding Steps (for CSR):
    - Read edgelist once to determine vertex degrees (CountDegrees)
    - Determine vertex offsets by a prefix sum (ParallelPrefixSum)
    - Allocate storage and set points according to offsets (GenIndex)
    - Copy edges into storage
  */
  void MakeCSR(const EdgeList &el, bool transpose, DestID_ ***index,
               DestID_ **neighs)
  {
    pvector<NodeID_> degrees = CountDegrees(el, transpose);
    pvector<SGOffset> offsets = ParallelPrefixSum(degrees);
    *neighs = new DestID_[offsets[num_nodes_]];
    *index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, *neighs);
#pragma omp parallel for
    for (auto it = el.begin(); it < el.end(); it++)
    {
      Edge e = *it;
      if (symmetrize_ || (!symmetrize_ && !transpose))
        (*neighs)[fetch_and_add(offsets[e.u], 1)] = e.v;
      if (symmetrize_ || (!symmetrize_ && transpose))
        (*neighs)[fetch_and_add(offsets[static_cast<NodeID_>(e.v)], 1)] =
            GetSource(e);
    }
  }

  CSRGraph<NodeID_, DestID_, invert> MakeGraphFromEL(EdgeList &el)
  {
    DestID_ **index = nullptr, **inv_index = nullptr;
    DestID_ *neighs = nullptr, *inv_neighs = nullptr;
    Timer t;
    t.Start();
    if (num_nodes_ == -1)
      num_nodes_ = FindMaxNodeID(el) + 1;
    if (needs_weights_)
      Generator<NodeID_, DestID_, WeightT_>::InsertWeights(el);
    if (in_place_)
    {
      MakeCSRInPlace(el, &index, &neighs, &inv_index, &inv_neighs);
    }
    else
    {
      MakeCSR(el, false, &index, &neighs);
      if (!symmetrize_ && invert)
      {
        MakeCSR(el, true, &inv_index, &inv_neighs);
      }
    }
    t.Stop();
    PrintTime("Build Time", t.Seconds());
    if (symmetrize_)
      return CSRGraph<NodeID_, DestID_, invert>(num_nodes_, index, neighs);
    else
      return CSRGraph<NodeID_, DestID_, invert>(num_nodes_, index, neighs,
                                                inv_index, inv_neighs);
  }

  CSRGraph<NodeID_, DestID_, invert> MakeGraph()
  {
    CSRGraph<NodeID_, DestID_, invert> g;
    { // extra scope to trigger earlier deletion of el (save memory)
      EdgeList el;
      if (cli_.filename() != "")
      {
        Reader<NodeID_, DestID_, WeightT_, invert> r(cli_.filename());
        if ((r.GetSuffix() == ".sg") || (r.GetSuffix() == ".wsg"))
        {
          return r.ReadSerializedGraph();
        }
        else
        {
          el = r.ReadFile(needs_weights_);
        }
      }
      else if (cli_.scale() != -1)
      {
        Generator<NodeID_, DestID_> gen(cli_.scale(), cli_.degree());
        el = gen.GenerateEL(cli_.uniform());
      }
      g = MakeGraphFromEL(el);
    }
    if (in_place_)
      return g;
    else
      return SquishGraph(g);
  }

  // Relabels (and rebuilds) graph by order of decreasing degree
  static CSRGraph<NodeID_, DestID_, invert> RelabelByDegree(
      const CSRGraph<NodeID_, DestID_, invert> &g)
  {
    if (g.directed())
    {
      std::cout << "Cannot relabel directed graph" << std::endl;
      std::exit(-11);
    }
    Timer t;
    t.Start();
    typedef std::pair<int64_t, NodeID_> degree_node_p;
    pvector<degree_node_p> degree_id_pairs(g.num_nodes());
#pragma omp parallel for
    for (NodeID_ n = 0; n < g.num_nodes(); n++)
      degree_id_pairs[n] = std::make_pair(g.out_degree(n), n);
    std::sort(degree_id_pairs.begin(), degree_id_pairs.end(),
              std::greater<degree_node_p>());
    pvector<NodeID_> degrees(g.num_nodes());
    pvector<NodeID_> new_ids(g.num_nodes());
#pragma omp parallel for
    for (NodeID_ n = 0; n < g.num_nodes(); n++)
    {
      degrees[n] = degree_id_pairs[n].first;
      new_ids[degree_id_pairs[n].second] = n;
    }
    pvector<SGOffset> offsets = ParallelPrefixSum(degrees);
    DestID_ *neighs = new DestID_[offsets[g.num_nodes()]];
    DestID_ **index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, neighs);
#pragma omp parallel for
    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      for (NodeID_ v : g.out_neigh(u))
        neighs[offsets[new_ids[u]]++] = new_ids[v];
      std::sort(index[new_ids[u]], index[new_ids[u] + 1]);
    }
    t.Stop();
    PrintTime("Relabel", t.Seconds());
    return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), index, neighs);
  }

  //........................... DBG ...........................
  static CSRGraph<NodeID_, DestID_, invert> generateDBGMapping(
      const CSRGraph<NodeID_, DestID_, invert> &g, bool useOutdeg, pvector<NodeID_> &new_ids, bool isWeighted = false)
  {
    typedef int32_t WeightT;
    typedef NodeWeight<NodeID_, WeightT> WNode;
    // typedef CSRGraph<NodeID_, WNode> WGraph;
    if (!g.directed())
    {
      std::cout << "Cannot relabel un-directed graph" << std::endl;
      std::exit(-11);
    }
    Timer t;
    t.Start();

    auto num_vertices = g.num_nodes();
    auto num_edges = g.num_edges();
    // vertex *origG    = GA.V;

    uint32_t avg_vertex = num_edges / num_vertices;
    const uint32_t &av = avg_vertex;

    uint32_t bucket_threshold[] = {av / 2, av, av * 2, av * 4, av * 8, av * 16, av * 32, av * 64, av * 128, av * 256, av * 512, static_cast<uint32_t>(-1)};
    int num_buckets = 11;
    if (num_buckets > 11)
    {
      // if you really want to increase the bucket count, add more thresholds to the bucket_threshold above.
      std::cout << "Unsupported bucket size: " << num_buckets << std::endl;
      assert(0);
    }
    bucket_threshold[num_buckets - 1] = static_cast<uint32_t>(-1);
    // pvector<NodeID_> new_ids(g.num_nodes());

    vector<uint32_t> bucket_vertices[num_buckets];
    const int num_threads = omp_get_max_threads();
    vector<uint32_t> local_buckets[num_threads][num_buckets];

    pvector<NodeID_> out_degrees(g.num_nodes());
    pvector<NodeID_> in_degrees(g.num_nodes());

    if (useOutdeg)
    {
      // This loop relies on a static scheduling
#pragma omp parallel for schedule(static)
      for (uint64_t i = 0; i < (unsigned)num_vertices; i++)
      {
        for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
        {
          // const uintE& count = origG[i].getOutDegree();
          const int64_t &count = g.out_degree(i);
          if (count <= bucket_threshold[j])
          {
            local_buckets[omp_get_thread_num()][j].push_back(i);
            break;
          }
        }
      }
    }
    else
    {
#pragma omp parallel for schedule(static)
      for (uint64_t i = 0; i < (unsigned)num_vertices; i++)
      {
        for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
        {
          const int64_t &count = g.in_degree(i);
          if (count <= bucket_threshold[j])
          {
            local_buckets[omp_get_thread_num()][j].push_back(i);
            break;
          }
        }
      }
    }

    int temp_k = 0;
    uint32_t start_k[num_threads][num_buckets];
    for (int32_t j = num_buckets - 1; j >= 0; j--)
    {
      for (int t = 0; t < num_threads; t++)
      {
        start_k[t][j] = temp_k;
        temp_k += local_buckets[t][j].size();
      }
    }

#pragma omp parallel for schedule(static)
    for (int t = 0; t < num_threads; t++)
    {
      for (int32_t j = num_buckets - 1; j >= 0; j--)
      {
        const vector<uint32_t> &current_bucket = local_buckets[t][j];
        int k = start_k[t][j];
        const size_t &size = current_bucket.size();
        for (uint32_t i = 0; i < size; i++)
        {
          new_ids[current_bucket[i]] = k++;
        }
      }
    }

    // uint32_t *local_buckets1;
    // local_buckets1=new uint32_t[num_buckets];
    // for(int32_t j = num_buckets -1; j >= 0; j--)
    // {
    // local_buckets1[j]=0;
    // }
    //     for (int32_t j = num_buckets - 1; j >= 0; j--)
    //       {
    // for (int t = 0; t < num_threads; t++)
    //       {

    //         local_buckets1[j]local_buckets[t][j].size();
    //       }}
    //   uint32_t new_start=0;
    //    uint32_t no_nodeleft=g.num_nodes();;
    // uint32_t depthsize=500;

    // uint32_t depth_size[num_buckets];
    // for(int32_t j = num_buckets -1; j >= 0; j--)
    // {
    //   // const vector<uint32_t> &current_bucket = local_buckets[j];
    //  double bucket_percentage= local_buckets1[j].size()/g.num_nodes();
    //  if(bucket_percentage<.1)
    //  {
    // depth_size[j]= 500;
    // }
    // else
    // {

    // depth_size[j]= 500*8;

    // }
    // }

    // uint32_t *bucket_offset;
    // bucket_offset=new uint32_t[num_buckets];

    // for(int32_t j = num_buckets -1; j >= 0; j--)
    // {
    // bucket_offset[j]=0;
    // }

    // cout<<"hi";
    // while(no_nodeleft)
    // {
    //     for(int32_t j = num_buckets - 1; j >= 0; j--)
    //     {  const vector<uint32_t> &current_bucket = local_buckets1[j];
    //       // if(current_bucket.empty())
    //       //   continue;
    //       for (int32_t k = 0; k<depth_size[j] && bucket_offset[j]<current_bucket.size(); k++)
    //       {
    //       //   if(!current_bucket.empty())
    //       // {
    //         new_ids[current_bucket[bucket_offset[j]]] = new_start++;
    //         no_nodeleft=no_nodeleft-1;

    //         bucket_offset[j]=bucket_offset[j]+1;

    //       }

    //     }
    // }

    // if(isWeighted)
    // {
    //   for(NodeID_ u = 0; u < g.num_nodes(); u++)
    //   {
    //     for(WNode wn: g.out_neigh(u))
    //     {
    //       cout << wn.w << endl;
    //     }
    //   }
    // }

    for (uint64_t i = 0; i < (uint64_t)num_threads; i++)
    {
      for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
      {
        local_buckets[i][j].clear();
      }
    }

#pragma omp parallel for
    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      out_degrees[new_ids[u]] = g.out_degree(u);
    }

#pragma omp parallel for
    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      in_degrees[new_ids[u]] = g.in_degree(u);
    }

    pvector<SGOffset> out_offsets = ParallelPrefixSum(out_degrees);
    DestID_ *out_neighs = new DestID_[out_offsets[g.num_nodes()]];
    DestID_ **out_index = CSRGraph<NodeID_, DestID_>::GenIndex(out_offsets, out_neighs);
    if (!isWeighted)
    {
#pragma omp parallel for
      for (NodeID_ u = 0; u < g.num_nodes(); u++)
      {
        for (NodeID_ v : g.out_neigh(u))
          out_neighs[out_offsets[new_ids[u]]++] = new_ids[v];
        std::sort(out_index[new_ids[u]], out_index[new_ids[u] + 1]);
      }
    }
    else
    {
#pragma omp parallel for
      for (NodeID_ u = 0; u < g.num_nodes(); u++)
      {
        for (WNode wn : g.out_neigh(u))
        {
          WNode new_node(new_ids[wn.v], wn.w);
          out_neighs[out_offsets[new_ids[u]]++] = new_node;
        }

        std::sort(out_index[new_ids[u]], out_index[new_ids[u] + 1]);
      }
    }
    pvector<SGOffset> in_offsets = ParallelPrefixSum(in_degrees);
    DestID_ *in_neighs = new DestID_[in_offsets[g.num_nodes()]];
    DestID_ **in_index = CSRGraph<NodeID_, DestID_>::GenIndex(in_offsets, in_neighs);
#pragma omp parallel for
    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      for (NodeID_ v : g.in_neigh(u))
        in_neighs[in_offsets[new_ids[u]]++] = new_ids[v];
      std::sort(in_index[new_ids[u]], in_index[new_ids[u] + 1]);
    }

    t.Stop();
    PrintTime("DBG Map Time", t.Seconds());

    if (isWeighted)
    {
      CSRGraph<NodeID_, DestID_, invert> g1(g.num_nodes(), out_index, out_neighs, in_index, in_neighs);
      // for(NodeID_ u = 0; u < g.num_nodes(); u++)
      // {
      //   for(WNode wn: g.out_neigh(u))
      //   {

      //     cout <<"old"<<new_ids[wn.v]<<"  "<< wn.w << endl;

      //   }
      //   for(WNode wn1: g1.out_neigh(new_ids[u]))
      //     {
      //       cout <<"Renamed"<<wn1.v<<"  "<< wn1.w << endl;

      //     }

      // }
      return g1;
    }
    return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), out_index, out_neighs, in_index, in_neighs);
  }
  //...........................................................
  //........................... DBG for undirected graphs...........................
  static CSRGraph<NodeID_, DestID_, invert> generateDBGMappingUndirected(
      const CSRGraph<NodeID_, DestID_, invert> &g, bool useOutdeg, pvector<NodeID_> &new_ids, bool isWeighted = false)
  {
    typedef int32_t WeightT;
    typedef NodeWeight<NodeID_, WeightT> WNode;
    // typedef CSRGraph<NodeID_, WNode> WGraph;
    if (g.directed())
    {
      std::cout << "Cannot relabel directed graph" << std::endl;
      std::exit(-11);
    }
    Timer t;
    t.Start();

    auto num_vertices = g.num_nodes();
    auto num_edges = g.num_edges();
    // vertex *origG    = GA.V;

    uint32_t avg_vertex = num_edges / num_vertices;
    const uint32_t &av = avg_vertex;

    uint32_t bucket_threshold[] = {av / 2, av, av * 2, av * 4, av * 8, av * 16, av * 32, av * 64, av * 128, av * 256, av * 512, static_cast<uint32_t>(-1)};
    int num_buckets = 8;
    if (num_buckets > 11)
    {
      // if you really want to increase the bucket count, add more thresholds to the bucket_threshold above.
      std::cout << "Unsupported bucket size: " << num_buckets << std::endl;
      assert(0);
    }
    bucket_threshold[num_buckets - 1] = static_cast<uint32_t>(-1);
    // pvector<NodeID_> new_ids(g.num_nodes());

    vector<uint32_t> bucket_vertices[num_buckets];
    const int num_threads = omp_get_max_threads();
    vector<uint32_t> local_buckets[num_threads][num_buckets];

    pvector<NodeID_> out_degrees(g.num_nodes());

    // This loop relies on a static scheduling
#pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < (unsigned)num_vertices; i++)
    {
      for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
      {
        // const uintE& count = origG[i].getOutDegree();
        const int64_t &count = g.out_degree(i);
        if (count <= bucket_threshold[j])
        {
          local_buckets[omp_get_thread_num()][j].push_back(i);
          break;
        }
      }
    }

    int temp_k = 0;
    uint32_t start_k[num_threads][num_buckets];
    for (int32_t j = num_buckets - 1; j >= 0; j--)
    {
      for (int t = 0; t < num_threads; t++)
      {
        start_k[t][j] = temp_k;
        temp_k += local_buckets[t][j].size();
      }
    }

#pragma omp parallel for schedule(static)
    for (int t = 0; t < num_threads; t++)
    {
      for (int32_t j = num_buckets - 1; j >= 0; j--)
      {
        const vector<uint32_t> &current_bucket = local_buckets[t][j];
        int k = start_k[t][j];
        const size_t &size = current_bucket.size();
        for (uint32_t i = 0; i < size; i++)
        {
          new_ids[current_bucket[i]] = k++;
        }
      }
    }

    for (uint64_t i = 0; i < (uint64_t)num_threads; i++)
    {
      for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
      {
        local_buckets[i][j].clear();
      }
    }

#pragma omp parallel for
    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      out_degrees[new_ids[u]] = g.out_degree(u);
    }

    pvector<SGOffset> out_offsets = ParallelPrefixSum(out_degrees);
    DestID_ *out_neighs = new DestID_[out_offsets[g.num_nodes()]];
    DestID_ **out_index = CSRGraph<NodeID_, DestID_>::GenIndex(out_offsets, out_neighs);
    if (!isWeighted)
    {
#pragma omp parallel for
      for (NodeID_ u = 0; u < g.num_nodes(); u++)
      {
        for (NodeID_ v : g.out_neigh(u))
          out_neighs[out_offsets[new_ids[u]]++] = new_ids[v];
        std::sort(out_index[new_ids[u]], out_index[new_ids[u] + 1]);
      }
    }
    else
    {
#pragma omp parallel for
      for (NodeID_ u = 0; u < g.num_nodes(); u++)
      {
        for (WNode wn : g.out_neigh(u))
        {
          WNode new_node(new_ids[wn.v], wn.w);
          out_neighs[out_offsets[new_ids[u]]++] = new_node;
        }

        std::sort(out_index[new_ids[u]], out_index[new_ids[u] + 1]);
      }
    }

    t.Stop();
    PrintTime("DBG Map Time", t.Seconds());

    return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), out_index, out_neighs);
  }
  //...........................................................

  //--------------------Our Ordering--------------------------

  static CSRGraph<NodeID_, DestID_, invert> generateNeighbourMapping(
      const CSRGraph<NodeID_, DestID_, invert> &g, bool useOutdeg, pvector<NodeID_> &new_ids, bool need_weights = false, bool isWeighted = false)
  {
    typedef int32_t WeightT;
    typedef NodeWeight<NodeID_, WeightT> WNode;
    // typedef CSRGraph<NodeID_, WNode> WGraph;
    if (!g.directed())
    {
      std::cout << "Cannot relabel un-directed graph" << std::endl;
      std::exit(-11);
    }
    Timer t;
    t.Start();

    auto num_vertices = g.num_nodes();
    auto num_edges = g.num_edges();
    // vertex *origG    = GA.V;

    uint32_t avg_vertex = num_edges / num_vertices;
    const uint32_t &av = avg_vertex;

    uint32_t bucket_threshold[] = {av / 2, av, av * 2, av * 4, av * 8, av * 16, av * 32, av * 64, av * 128, av * 256, av * 512, static_cast<uint32_t>(-1)};
    // making sure we use 8 buckets now
    int num_buckets = 8;
    if (num_buckets > 11)
    {
      // if you really want to increase the bucket count, add more thresholds to the bucket_threshold above.
      std::cout << "Unsupported bucket size: " << num_buckets << std::endl;
      assert(0);
    }
    bucket_threshold[num_buckets - 1] = static_cast<uint32_t>(-1);
    // pvector<NodeID_> new_ids(g.num_nodes(), -1);

    vector<uint32_t> bucket_vertices[num_buckets];
    // const int num_threads = omp_get_max_threads();
    vector<uint32_t> local_buckets[num_buckets];

    pvector<NodeID_> out_degrees(g.num_nodes());
    pvector<NodeID_> in_degrees(g.num_nodes());

    list<NodeID_> queue;
    bool *visited = new bool[g.num_nodes()];
#pragma omp parallel for
    for (int i = 0; i < g.num_nodes(); i++)
      visited[i] = false;
    // NodeID_ start_vertex=5;
    // queue.push_back(start_vertex);
    // visited[start_vertex]=true;

    // NodeID_ s=start_vertex;
    NodeID_ start_node = 0;
    int64_t max_degree = 0;

    if (useOutdeg){

      //start from max in degree

#pragma omp parallel for reduction(max \
                                   : max_degree, start_node)
      for (NodeID_ i = 0; i < g.num_nodes(); i++)
      {
        if (g.in_degree(i) > max_degree)
        {
          max_degree = g.in_degree(i);
          start_node = i;
        }
      }

      list<NodeID_> queue1;
      queue1.push_back(start_node);
      visited[start_node] = true;
      for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
      {
        const int64_t &count = g.out_degree(start_node);
        if (count <= bucket_threshold[j])
        {
          local_buckets[j].push_back(start_node);
          break;
        }
      }
      NodeID_ s1;

      //#pragma omp parallel for
      //for(int i=0; i>0;i++)
      while (!queue1.empty())
      {

        int temp_size = queue1.size();
        //cout<<"Temp_size of queue "<<temp_size<<endl;
        //if(!queue1.empty()){  //if block starts
        //#pragma omp parallel for
        for(int i=0;i<temp_size;i++){ //for loop starts

        s1 = queue1.front();
        queue1.pop_front();
      

        int test_var = 0;
        
        for (auto v = g.in_neigh(s1).begin(); v != g.in_neigh(s1).end(); v++)
        {
          // #pragma omp single nowait
          // {
          if (!visited[*v])
          { 
            //cout<<"Unvisited node found"<<endl;
            //#pragma omp critical 
            //{  //critical section starts
            visited[*v] = true;
            // #pragma omp critical
            queue1.push_back(*v);
             // } // critical section ends
            // #pragma omp parallel for schedule(static)
            for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
            {
              const int64_t &count = g.out_degree(*v);
              if (count <= bucket_threshold[j])
              {
                //cout<<"Node added to bucket"<<endl;
                local_buckets[j].push_back(*v);
                break;
              }
            }

            test_var++;
          }
          
        }
        //cout<<"Added "<<test_var<<" nodes in this iteration"<<endl;

      } //for loop ends
      //} // if block ends
      //#pragma omp barrier
      }


      // Uptil now, only 1 element in the queue

      // This loop relies on a static scheduling
      //cout<<"Static scheduling now"<<endl;
      for (NodeID_ u = 0; u < g.num_nodes(); u++)
      {
        if (!visited[u])
        {
          //cout<<"Unvisted node made the source"<<endl;
          list<NodeID_> queue;
          queue.push_back(u);
          visited[u] = true;
          for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
          {
            const int64_t &count = g.out_degree(u);
            if (count <= bucket_threshold[j])
            {
              //cout<<"Unvisited node found #1"<<endl;
              local_buckets[j].push_back(u);
              break;
            }
          }
          NodeID_ s;
          while (!queue.empty())
          {
            s = queue.front();
            //cout << s << " ";
            queue.pop_front();
            // #pragma omp parallel
            // {
            // for (auto v = g.out_neigh(s).begin(); v != g.out_neigh(s).end(); v++)
            //int test_var=0;
            for (auto v = g.in_neigh(s).begin(); v != g.in_neigh(s).end(); v++)
            {
              //cout<<"Checking neighbours"<<endl;
              // #pragma omp single nowait
              // {
              if (!visited[*v])
              {
                //test_var++;
                //cout<<"Unvisited node found"<<endl;
                visited[*v] = true;
                // #pragma omp critical
                queue.push_back(*v);

                // #pragma omp parallel for schedule(static)
                for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
                {
                  const int64_t &count = g.out_degree(*v);
                  if (count <= bucket_threshold[j])
                  {
                    //cout<<"Node added to bucket"<<endl;
                    local_buckets[j].push_back(*v);
                    break;
                  }
                }
              }
              
            }
            //cout<<"Number of nodes in this iteration "<<test_var<<endl;
          }
          // }
          // }
        }
      }
    }

    else
    {
//Use  out degee for bfs


//cout<<"Using out-degrees now"<<endl;

#pragma omp parallel for reduction(max \
                                   : max_degree, start_node)
      for (NodeID_ i = 0; i < g.num_nodes(); i++)
      {
        if (g.out_degree(i) > max_degree)
        {
          max_degree = g.out_degree(i);
          start_node = i;
        }
      }

      list<NodeID_> queue1;
      queue1.push_back(start_node);
      visited[start_node] = true;
      for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
      {
        const int64_t &count = g.out_degree(start_node);
        if (count <= bucket_threshold[j])
        {
          local_buckets[j].push_back(start_node);
          break;
        }
      }
      NodeID_ s1;
      while (!queue1.empty())
      {
        s1 = queue1.front();
        // cout << s << " ";
        queue1.pop_front();
        // #pragma omp parallel
        // {
        // for (auto v = g.out_neigh(s).begin(); v != g.out_neigh(s).end(); v++)
        for (auto v = g.in_neigh(s1).begin(); v != g.in_neigh(s1).end(); v++)
        {
          // #pragma omp single nowait
          // {
          if (!visited[*v])
          {
            //cout<<"Unvisited node found"<<endl;
            visited[*v] = true;
            queue.push_back(*v);

            // #pragma omp parallel for schedule(static)
            for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
            {
              const int64_t &count = g.out_degree(*v);
              if (count <= bucket_threshold[j])
              {
                local_buckets[j].push_back(*v);
                break;
              }
            }
          }
        }
      }

      for (NodeID_ u = 0; u < g.num_nodes(); u++)
      {

        if (!visited[u])
        {
          list<NodeID_> queue;
          queue.push_back(u);
          visited[u] = true;
          for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
          {
            const int64_t &count = g.in_degree(u);
            if (count <= bucket_threshold[j])
            {
              local_buckets[j].push_back(u);
              break;
            }
          }
          NodeID_ s;

          while (!queue.empty())
          {
            s = queue.front();
            // cout << s << " ";
            queue.pop_front();
            // for (NodeID_ v : g.in_neigh(s))
            for (NodeID_ v : g.out_neigh(s))
            {
              if (!visited[v])
              {
                visited[v] = true;
                queue.push_back(v);

                // #pragma omp parallel for schedule(static)
                for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
                {
                  const int64_t &count = g.in_degree(v);
                  if (count <= bucket_threshold[j])
                  {
                    local_buckets[j].push_back(v);
                    break;
                  }
                }
              }
            }
          }
        }
      }
    }

    //here we have to apply reorder logic for buckets where we can set w parameter
    int temp_k = 0;
    uint32_t start_k[num_buckets];
    for (int32_t j = num_buckets - 1; j >= 0; j--)
    {
      start_k[j] = temp_k;
      temp_k += local_buckets[j].size();
    }

    for (int32_t j = num_buckets - 1; j >= 0; j--)
    {
      const vector<uint32_t> &current_bucket = local_buckets[j];
      int k = start_k[j];
      const size_t &size = current_bucket.size();
      for (uint32_t i = 0; i < size; i++)
      {
        new_ids[current_bucket[i]] = k++;
      }
    }

    for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
    {
      local_buckets[j].clear();
    }

    // for (NodeID_ u = 0; u < g.num_nodes(); u++)
    // {
    //   if (new_ids[u] == -1)
    //   {
    //     cout << "Node id not assigned to: " << u << endl;
    //   }
    // }

    // pvector<WeightT_> new_weights(g.nu)

#pragma omp parallel for
    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      out_degrees[new_ids[u]] = g.out_degree(u);
    }

#pragma omp parallel for
    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      in_degrees[new_ids[u]] = g.in_degree(u);
    }
    pvector<SGOffset> out_offsets = ParallelPrefixSum(out_degrees);
    DestID_ *out_neighs = new DestID_[out_offsets[g.num_nodes()]];
    DestID_ **out_index = CSRGraph<NodeID_, DestID_>::GenIndex(out_offsets, out_neighs);
    if (!isWeighted)
    {
#pragma omp parallel for
      for (NodeID_ u = 0; u < g.num_nodes(); u++)
      {
        for (NodeID_ v : g.out_neigh(u))
          out_neighs[out_offsets[new_ids[u]]++] = new_ids[v];
        std::sort(out_index[new_ids[u]], out_index[new_ids[u] + 1]);
      }
    }
    else
    {
#pragma omp parallel for
      for (NodeID_ u = 0; u < g.num_nodes(); u++)
      {
        for (WNode wn : g.out_neigh(u))
        {
          WNode new_node(new_ids[wn.v], wn.w);
          out_neighs[out_offsets[new_ids[u]]++] = new_node;
        }

        std::sort(out_index[new_ids[u]], out_index[new_ids[u] + 1]);
      }
    }

    pvector<SGOffset> in_offsets = ParallelPrefixSum(in_degrees);
    DestID_ *in_neighs = new DestID_[in_offsets[g.num_nodes()]];
    DestID_ **in_index = CSRGraph<NodeID_, DestID_>::GenIndex(in_offsets, in_neighs);
#pragma omp parallel for
    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      for (NodeID_ v : g.in_neigh(u))
        in_neighs[in_offsets[new_ids[u]]++] = new_ids[v];
      std::sort(in_index[new_ids[u]], in_index[new_ids[u] + 1]);
    }

    // assert(new_ids[2721009] == 2218313);
    t.Stop();
    PrintTime("Neighbourhood Map Time", t.Seconds());

    return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), out_index, out_neighs, in_index, in_neighs);
  }

  //................... our reordering parallel ..........................
  static CSRGraph<NodeID_, DestID_, invert> generateParallelNeighbourMapping(
      const CSRGraph<NodeID_, DestID_, invert> &g, bool useOutdeg, pvector<NodeID_> &new_ids, bool isWeighted = false)
  {
    typedef int32_t WeightT;
    typedef NodeWeight<NodeID_, WeightT> WNode;
    // typedef CSRGraph<NodeID_, WNode> WGraph;
    if (!g.directed())
    {
      std::cout << "Cannot relabel un-directed graph" << std::endl;
      std::exit(-11);
    }
    Timer t;
    t.Start();

    auto num_vertices = g.num_nodes();
    auto num_edges = g.num_edges();
    // vertex *origG    = GA.V;

    uint32_t avg_vertex = num_edges / num_vertices;
    const uint32_t &av = avg_vertex;

    uint32_t bucket_threshold[] = {av / 2, av, av * 2, av * 4, av * 8, av * 16, av * 32, av * 64, av * 128, av * 256, av * 512, static_cast<uint32_t>(-1)};
    // making sure we use 8 buckets now
    int num_buckets = 8;
    if (num_buckets > 11)
    {
      // if you really want to increase the bucket count, add more thresholds to the bucket_threshold above.
      std::cout << "Unsupported bucket size: " << num_buckets << std::endl;
      assert(0);
    }
    bucket_threshold[num_buckets - 1] = static_cast<uint32_t>(-1);
    // pvector<NodeID_> new_ids(g.num_nodes(), -1);

    vector<uint32_t> bucket_vertices[num_buckets];
    // const int num_threads = omp_get_max_threads();
    vector<uint32_t> local_buckets[num_buckets];

    pvector<NodeID_> out_degrees(g.num_nodes());
    pvector<NodeID_> in_degrees(g.num_nodes());

    list<NodeID_> queue;
    pvector<bool> visited(g.num_nodes());
#pragma omp parallel for
    for (int i = 0; i < g.num_nodes(); i++)
      visited[i] = false;

    if (useOutdeg)
    {
      for (NodeID_ u = 0; u < g.num_nodes(); u++)
      {
        if (!visited[u])
        {
          visited[u] = true;

          for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
          {
            const int64_t &count = g.out_degree(u);
            if (count <= bucket_threshold[j])
            {
              local_buckets[j].push_back(u);
              break;
            }
          }

          SlidingQueue<NodeID_> queue(g.num_nodes());
          queue.push_back(u);
          queue.slide_window();
          Bitmap curr(g.num_nodes());
          curr.reset();
          Bitmap front(g.num_nodes());
          front.reset();

#pragma omp parallel
          {
            QueueBuffer<NodeID_> lqueue(queue);
#pragma omp for
            for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++)
            {
              NodeID_ u = *q_iter;
              for (NodeID_ v : g.out_neigh(u))
              {
                bool curr_val = visited[v];
                if (curr_val == 0)
                {
                  if (compare_and_swap(visited[v], curr_val, true))
                  {
                    lqueue.push_back(v);
                    for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
                    {
                      const int64_t &count = g.out_degree(v);
                      if (count <= bucket_threshold[j])
                      {
                        local_buckets[j].push_back(v);
                        break;
                      }
                    }
                  }
                }
              }
            }
            lqueue.flush();
          }
        }
      }
    }

    else
    {
      for (NodeID_ u = 0; u < g.num_nodes(); u++)
      {
        if (!visited[u])
        {
          visited[u] = true;

          for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
          {
            const int64_t &count = g.in_degree(u);
            if (count <= bucket_threshold[j])
            {
              local_buckets[j].push_back(u);
              break;
            }
          }

          SlidingQueue<NodeID_> queue(g.num_nodes());
          queue.push_back(u);
          queue.slide_window();
          Bitmap curr(g.num_nodes());
          curr.reset();
          Bitmap front(g.num_nodes());
          front.reset();

#pragma omp parallel
          {
            QueueBuffer<NodeID_> lqueue(queue);
#pragma omp for
            for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++)
            {
              NodeID_ u = *q_iter;
              for (NodeID_ v : g.in_neigh(u))
              {
                bool curr_val = visited[v];
                if (curr_val == 0)
                {
                  if (compare_and_swap(visited[v], curr_val, true))
                  {
                    lqueue.push_back(v);
                    for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
                    {
                      const int64_t &count = g.in_degree(v);
                      if (count <= bucket_threshold[j])
                      {
                        local_buckets[j].push_back(v);
                        break;
                      }
                    }
                  }
                }
              }
            }
            lqueue.flush();
          }
        }
      }
    }
    //here we have to apply reorder logic for buckets where we can set w parameter
    int temp_k = 0;
    uint32_t start_k[num_buckets];
    for (int32_t j = num_buckets - 1; j >= 0; j--)
    {
      start_k[j] = temp_k;
      temp_k += local_buckets[j].size();
    }

    for (int32_t j = num_buckets - 1; j >= 0; j--)
    {
      const vector<uint32_t> &current_bucket = local_buckets[j];
      int k = start_k[j];
      const size_t &size = current_bucket.size();
      for (uint32_t i = 0; i < size; i++)
      {
        new_ids[current_bucket[i]] = k++;
      }
    }

    for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
    {
      local_buckets[j].clear();
    }

    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      if (new_ids[u] == -1)
      {
        cout << "Node id not assigned to: " << u << endl;
      }
    }

#pragma omp parallel for
    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      out_degrees[new_ids[u]] = g.out_degree(u);
    }

#pragma omp parallel for
    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      in_degrees[new_ids[u]] = g.in_degree(u);
    }

    pvector<SGOffset> out_offsets = ParallelPrefixSum(out_degrees);
    DestID_ *out_neighs = new DestID_[out_offsets[g.num_nodes()]];
    DestID_ **out_index = CSRGraph<NodeID_, DestID_>::GenIndex(out_offsets, out_neighs);
    if (!isWeighted)
    {
#pragma omp parallel for
      for (NodeID_ u = 0; u < g.num_nodes(); u++)
      {
        for (NodeID_ v : g.out_neigh(u))
          out_neighs[out_offsets[new_ids[u]]++] = new_ids[v];
        std::sort(out_index[new_ids[u]], out_index[new_ids[u] + 1]);
      }
    }
    else
    {
#pragma omp parallel for
      for (NodeID_ u = 0; u < g.num_nodes(); u++)
      {
        for (WNode wn : g.out_neigh(u))
        {
          WNode new_node(new_ids[wn.v], wn.w);
          out_neighs[out_offsets[new_ids[u]]++] = new_node;
        }

        std::sort(out_index[new_ids[u]], out_index[new_ids[u] + 1]);
      }
    }

    pvector<SGOffset> in_offsets = ParallelPrefixSum(in_degrees);
    DestID_ *in_neighs = new DestID_[in_offsets[g.num_nodes()]];
    DestID_ **in_index = CSRGraph<NodeID_, DestID_>::GenIndex(in_offsets, in_neighs);
#pragma omp parallel for
    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      for (NodeID_ v : g.in_neigh(u))
        in_neighs[in_offsets[new_ids[u]]++] = new_ids[v];
      std::sort(in_index[new_ids[u]], in_index[new_ids[u] + 1]);
    }

    t.Stop();
    PrintTime("Neighbourhood Map Time", t.Seconds());

    return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), out_index, out_neighs, in_index, in_neighs);
  }

  //...........................................................
  //-----------Numbering Node------------------

  static CSRGraph<NodeID_, DestID_, invert> generateNumberNodeNeighbourMapping(
      const CSRGraph<NodeID_, DestID_, invert> &g, bool useOutdeg, pvector<NodeID_> &new_ids, bool isWeighted = false)
  {
    typedef int32_t WeightT;
    typedef NodeWeight<NodeID_, WeightT> WNode;
    // typedef CSRGraph<NodeID_, WNode> WGraph;
    if (!g.directed())
    {
      std::cout << "Cannot relabel un-directed graph" << std::endl;
      std::exit(-11);
    }
    Timer t;
    t.Start();

    auto num_vertices = g.num_nodes();
    auto num_edges = g.num_edges();
    // vertex *origG    = GA.V;

    uint32_t avg_vertex = num_edges / num_vertices;
    const uint32_t &av = avg_vertex;

    uint32_t bucket_threshold[] = {av / 2, av, av * 2, av * 4, av * 8, av * 16, av * 32, av * 64, av * 128, av * 256, av * 512, static_cast<uint32_t>(-1)};
    // making sure we use 8 buckets now
    int num_buckets = 8;
    if (num_buckets > 11)
    {
      // if you really want to increase the bucket count, add more thresholds to the bucket_threshold above.
      std::cout << "Unsupported bucket size: " << num_buckets << std::endl;
      assert(0);
    }
    bucket_threshold[num_buckets - 1] = static_cast<uint32_t>(-1);
    // pvector<NodeID_> new_ids(g.num_nodes(), -1);

    vector<uint32_t> bucket_vertices[num_buckets];
    // const int num_threads = omp_get_max_threads();
    vector<uint32_t> local_buckets[num_buckets];

    pvector<NodeID_> out_degrees(g.num_nodes());
    pvector<NodeID_> in_degrees(g.num_nodes());

    list<NodeID_> queue;
    bool *visited = new bool[g.num_nodes()];
#pragma omp parallel for
    for (int i = 0; i < g.num_nodes(); i++)
      visited[i] = false;
    // NodeID_ start_vertex=5;
    // queue.push_back(start_vertex);
    // visited[start_vertex]=true;

    // NodeID_ s=start_vertex;
    NodeID_ start_node = 0;
    int64_t max_degree = 0;

    cout << "starting traversal";
    if (useOutdeg)
    {

      //start from max in degree

#pragma omp parallel for reduction(max \
                                   : max_degree, start_node)
      for (NodeID_ i = 0; i < g.num_nodes(); i++)
      {
        if (g.in_degree(i) > max_degree)
        {
          max_degree = g.in_degree(i);
          start_node = i;
        }
      }

      list<NodeID_> queue1;
      queue1.push_back(start_node);
      visited[start_node] = true;
      for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
      {
        const int64_t &count = g.out_degree(start_node);
        if (count <= bucket_threshold[j])
        {
          local_buckets[j].push_back(start_node);
          break;
        }
      }
      NodeID_ s1;
      while (!queue1.empty())
      {
        s1 = queue1.front();
        // cout << s << " ";
        queue1.pop_front();
        // #pragma omp parallel
        // {
        // for (auto v = g.out_neigh(s).begin(); v != g.out_neigh(s).end(); v++)
        for (auto v = g.in_neigh(s1).begin(); v != g.in_neigh(s1).end(); v++)
        {
          // #pragma omp single nowait
          // {
          if (!visited[*v])
          {
            visited[*v] = true;
            // #pragma omp critical
            queue.push_back(*v);

            // #pragma omp parallel for schedule(static)
            for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
            {
              const int64_t &count = g.out_degree(*v);
              if (count <= bucket_threshold[j])
              {
                local_buckets[j].push_back(*v);
                break;
              }
            }
          }
        }
      }

      // This loop relies on a static scheduling
      for (NodeID_ u = 0; u < g.num_nodes(); u++)
      {
        if (!visited[u])
        {
          list<NodeID_> queue;
          queue.push_back(u);
          visited[u] = true;
          for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
          {
            const int64_t &count = g.out_degree(u);
            if (count <= bucket_threshold[j])
            {
              local_buckets[j].push_back(u);
              break;
            }
          }
          NodeID_ s;
          while (!queue.empty())
          {
            s = queue.front();
            // cout << s << " ";
            queue.pop_front();
            // #pragma omp parallel
            // {
            // for (auto v = g.out_neigh(s).begin(); v != g.out_neigh(s).end(); v++)
            for (auto v = g.in_neigh(s).begin(); v != g.in_neigh(s).end(); v++)
            {
              // #pragma omp single nowait
              // {
              if (!visited[*v])
              {
                visited[*v] = true;
                // #pragma omp critical
                queue.push_back(*v);

                // #pragma omp parallel for schedule(static)
                for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
                {
                  const int64_t &count = g.out_degree(*v);
                  if (count <= bucket_threshold[j])
                  {
                    local_buckets[j].push_back(*v);
                    break;
                  }
                }
              }
            }
          }
          // }
          // }
        }
      }
    }

    else
    {
//Use  out degee for bfs
#pragma omp parallel for reduction(max \
                                   : max_degree, start_node)
      for (NodeID_ i = 0; i < g.num_nodes(); i++)
      {
        if (g.out_degree(i) > max_degree)
        {
          max_degree = g.out_degree(i);
          start_node = i;
        }
      }

      list<NodeID_> queue1;
      queue1.push_back(start_node);
      visited[start_node] = true;
      for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
      {
        const int64_t &count = g.out_degree(start_node);
        if (count <= bucket_threshold[j])
        {
          local_buckets[j].push_back(start_node);
          break;
        }
      }
      NodeID_ s1;
      while (!queue1.empty())
      {
        s1 = queue1.front();
        // cout << s << " ";
        queue1.pop_front();
        // #pragma omp parallel
        // {
        // for (auto v = g.out_neigh(s).begin(); v != g.out_neigh(s).end(); v++)
        for (auto v = g.in_neigh(s1).begin(); v != g.in_neigh(s1).end(); v++)
        {
          // #pragma omp single nowait
          // {
          if (!visited[*v])
          {
            visited[*v] = true;
            // #pragma omp critical
            queue.push_back(*v);

            // #pragma omp parallel for schedule(static)
            for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
            {
              const int64_t &count = g.out_degree(*v);
              if (count <= bucket_threshold[j])
              {
                local_buckets[j].push_back(*v);
                break;
              }
            }
          }
        }
      }

      for (NodeID_ u = 0; u < g.num_nodes(); u++)
      {

        if (!visited[u])
        {
          list<NodeID_> queue;
          queue.push_back(u);
          visited[u] = true;
          for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
          {
            const int64_t &count = g.in_degree(u);
            if (count <= bucket_threshold[j])
            {
              local_buckets[j].push_back(u);
              break;
            }
          }
          NodeID_ s;

          while (!queue.empty())
          {
            s = queue.front();
            // cout << s << " ";
            queue.pop_front();
            // for (NodeID_ v : g.in_neigh(s))
            for (NodeID_ v : g.out_neigh(s))
            {
              if (!visited[v])
              {
                visited[v] = true;
                queue.push_back(v);

                // #pragma omp parallel for schedule(static)
                for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
                {
                  const int64_t &count = g.in_degree(v);
                  if (count <= bucket_threshold[j])
                  {
                    local_buckets[j].push_back(v);
                    break;
                  }
                }
              }
            }
          }
        }
      }
    }
    cout << "hi1";
    //here we have to apply reorder logic for buckets where we can set w parameter
    uint32_t temp_k = 0;
    uint32_t start_k[num_buckets];
    for (int32_t j = num_buckets - 1; j >= 0; j--)
    {
      start_k[j] = temp_k;
      temp_k += local_buckets[j].size();
      cout << "Bucket size of " << j << "is= " << local_buckets[j].size() << " Cumulative bucket size: " << temp_k << endl;
      
    }

    for (int32_t j = num_buckets - 1; j >= num_buckets - 2; j--)
    {
      const vector<uint32_t> &current_bucket = local_buckets[j];
      int k = start_k[j];
      const size_t &size = current_bucket.size();
      for (uint32_t i = 0; i < size; i++)
      {
        new_ids[current_bucket[i]] = k++;
      }
    }

    uint32_t new_start = local_buckets[num_buckets - 1].size() + local_buckets[num_buckets - 2].size();
    uint32_t no_nodeleft = g.num_nodes() - local_buckets[num_buckets - 1].size() - local_buckets[num_buckets - 2].size();
    // uint32_t depthsize=500;
    uint32_t last_2_bucket = local_buckets[0].size() + local_buckets[1].size();
    uint32_t *bucket_offset;
    bucket_offset = new uint32_t[num_buckets - 2];

    for (int32_t j = num_buckets - 3; j >= 0; j--)
    {
      bucket_offset[j] = 0;
    }

    // cout<<"hi";
    // while(no_nodeleft)
    // {
    //     for(int32_t j = num_buckets - 3; j >= 0; j--)
    //     {  const vector<uint32_t> &current_bucket = local_buckets[j];
    //       // if(current_bucket.empty())
    //       //   continue;
    //       for (int32_t k = 0; k<depthsize && bucket_offset[j]<current_bucket.size(); k++)
    //       {
    //       //   if(!current_bucket.empty())
    //       // {
    //         new_ids[current_bucket[bucket_offset[j]]] = new_start++;
    //         no_nodeleft=no_nodeleft-1;

    //         bucket_offset[j]=bucket_offset[j]+1;

    //       }

    //     }
    // }

    uint32_t depthsize[num_buckets];
    for (int32_t j = num_buckets - 1; j >= 0; j--)
    {
      // const vector<uint32_t> &current_bucket = local_buckets[j];
      double bucket_percentage = local_buckets[j].size() / g.num_nodes();
      if (bucket_percentage < .1)
      {
        depthsize[j] = local_buckets[j].size() / 10;
      }
      else
      {

        depthsize[j] = 500 * 8;
      }
    }

    while (no_nodeleft - last_2_bucket)
    {
      for (int32_t j = num_buckets - 3; j >= 2; j--)
      {
        const vector<uint32_t> &current_bucket = local_buckets[j];
        // if(current_bucket.empty())
        //   continue;
        for (int32_t k = 0; k < depthsize[j] && bucket_offset[j] < current_bucket.size(); k++)
        {
          //   if(!current_bucket.empty())
          // {
          new_ids[current_bucket[bucket_offset[j]]] = new_start++;
          no_nodeleft = no_nodeleft - 1;

          bucket_offset[j] = bucket_offset[j] + 1;
        }
      }
    }

    for (int32_t j = 1; j >= 0; j--)
    {
      const vector<uint32_t> &current_bucket = local_buckets[j];
      // if(current_bucket.empty())
      //   continue;
      for (int32_t k = 0; k < current_bucket.size(); k++)
      {

        new_ids[current_bucket[k]] = new_start++;
      }
    }

    for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
    {
      local_buckets[j].clear();
    }

    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      if (new_ids[u] == -1)
      {
        cout << "Node id not assigned to: " << u << endl;
      }
    }

#pragma omp parallel for
    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      out_degrees[new_ids[u]] = g.out_degree(u);
    }

#pragma omp parallel for
    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      in_degrees[new_ids[u]] = g.in_degree(u);
    }

    pvector<SGOffset> out_offsets = ParallelPrefixSum(out_degrees);
    DestID_ *out_neighs = new DestID_[out_offsets[g.num_nodes()]];
    DestID_ **out_index = CSRGraph<NodeID_, DestID_>::GenIndex(out_offsets, out_neighs);

    if (!isWeighted)
    {
#pragma omp parallel for
      for (NodeID_ u = 0; u < g.num_nodes(); u++)
      {
        for (NodeID_ v : g.out_neigh(u))
          out_neighs[out_offsets[new_ids[u]]++] = new_ids[v];
        std::sort(out_index[new_ids[u]], out_index[new_ids[u] + 1]);
      }
    }
    else
    {
#pragma omp parallel for
      for (NodeID_ u = 0; u < g.num_nodes(); u++)
      {
        for (WNode wn : g.out_neigh(u))
        {
          WNode new_node(new_ids[wn.v], wn.w);
          out_neighs[out_offsets[new_ids[u]]++] = new_node;
        }

        std::sort(out_index[new_ids[u]], out_index[new_ids[u] + 1]);
      }
    }

    pvector<SGOffset> in_offsets = ParallelPrefixSum(in_degrees);
    DestID_ *in_neighs = new DestID_[in_offsets[g.num_nodes()]];
    DestID_ **in_index = CSRGraph<NodeID_, DestID_>::GenIndex(in_offsets, in_neighs);
#pragma omp parallel for
    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      for (NodeID_ v : g.in_neigh(u))
        in_neighs[in_offsets[new_ids[u]]++] = new_ids[v];
      std::sort(in_index[new_ids[u]], in_index[new_ids[u] + 1]);
    }

    t.Stop();
    PrintTime("Neighbourhood Map Time", t.Seconds());

    return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), out_index, out_neighs, in_index, in_neighs);
  }

  //node number type2

  static CSRGraph<NodeID_, DestID_, invert> generateNumberNodeNeighbourMapping2(
      const CSRGraph<NodeID_, DestID_, invert> &g, bool useOutdeg, pvector<NodeID_> &new_ids, bool isWeighted = false)
  {
    typedef int32_t WeightT;
    typedef NodeWeight<NodeID_, WeightT> WNode;
    // typedef CSRGraph<NodeID_, WNode> WGraph;
    if (!g.directed())
    {
      std::cout << "Cannot relabel un-directed graph" << std::endl;
      std::exit(-11);
    }
    Timer t;
    t.Start();

    auto num_vertices = g.num_nodes();
    auto num_edges = g.num_edges();
    // vertex *origG    = GA.V;

    uint32_t avg_vertex = num_edges / num_vertices;
    const uint32_t &av = avg_vertex;

    uint32_t bucket_threshold[] = {av / 2, av, av * 2, av * 4, av * 8, av * 16, av * 32, av * 64, av * 128, av * 256, av * 512, static_cast<uint32_t>(-1)};
    // making sure we use 8 buckets now
    int num_buckets = 11;
    if (num_buckets > 11)
    {
      // if you really want to increase the bucket count, add more thresholds to the bucket_threshold above.
      std::cout << "Unsupported bucket size: " << num_buckets << std::endl;
      assert(0);
    }
    bucket_threshold[num_buckets - 1] = static_cast<uint32_t>(-1);
    // pvector<NodeID_> new_ids(g.num_nodes(), -1);

    vector<uint32_t> bucket_vertices[num_buckets];
    // const int num_threads = omp_get_max_threads();
    vector<uint32_t> local_buckets[num_buckets];

    pvector<NodeID_> out_degrees(g.num_nodes());
    pvector<NodeID_> in_degrees(g.num_nodes());

    list<NodeID_> queue;
    bool *visited = new bool[g.num_nodes()];
#pragma omp parallel for
    for (int i = 0; i < g.num_nodes(); i++)
      visited[i] = false;
    // NodeID_ start_vertex=5;
    // queue.push_back(start_vertex);
    // visited[start_vertex]=true;

    // NodeID_ s=start_vertex;
    NodeID_ start_node = 0;
    int64_t max_degree = 0;

    cout << "starting traversal";
    if (useOutdeg)
    {

      //start from max in degree

#pragma omp parallel for reduction(max \
                                   : max_degree, start_node)
      for (NodeID_ i = 0; i < g.num_nodes(); i++)
      {
        if (g.in_degree(i) > max_degree)
        {
          max_degree = g.in_degree(i);
          start_node = i;
        }
      }

      list<NodeID_> queue1;
      queue1.push_back(start_node);
      visited[start_node] = true;
      for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
      {
        const int64_t &count = g.out_degree(start_node);
        if (count <= bucket_threshold[j])
        {
          local_buckets[j].push_back(start_node);
          break;
        }
      }
      NodeID_ s1;
      while (!queue1.empty())
      {
        s1 = queue1.front();
        // cout << s << " ";
        queue1.pop_front();
        // #pragma omp parallel
        // {
        // for (auto v = g.out_neigh(s).begin(); v != g.out_neigh(s).end(); v++)
        for (auto v = g.in_neigh(s1).begin(); v != g.in_neigh(s1).end(); v++)
        {
          // #pragma omp single nowait
          // {
          if (!visited[*v])
          {
            visited[*v] = true;
            // #pragma omp critical
            queue.push_back(*v);

            // #pragma omp parallel for schedule(static)
            for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
            {
              const int64_t &count = g.out_degree(*v);
              if (count <= bucket_threshold[j])
              {
                local_buckets[j].push_back(*v);
                break;
              }
            }
          }
        }
      }

      // This loop relies on a static scheduling
      for (NodeID_ u = 0; u < g.num_nodes(); u++)
      {
        if (!visited[u])
        {
          list<NodeID_> queue;
          queue.push_back(u);
          visited[u] = true;
          for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
          {
            const int64_t &count = g.out_degree(u);
            if (count <= bucket_threshold[j])
            {
              local_buckets[j].push_back(u);
              break;
            }
          }
          NodeID_ s;
          while (!queue.empty())
          {
            s = queue.front();
            // cout << s << " ";
            queue.pop_front();
            // #pragma omp parallel
            // {
            // for (auto v = g.out_neigh(s).begin(); v != g.out_neigh(s).end(); v++)
            for (auto v = g.in_neigh(s).begin(); v != g.in_neigh(s).end(); v++)
            {
              // #pragma omp single nowait
              // {
              if (!visited[*v])
              {
                visited[*v] = true;
                // #pragma omp critical
                queue.push_back(*v);

                // #pragma omp parallel for schedule(static)
                for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
                {
                  const int64_t &count = g.out_degree(*v);
                  if (count <= bucket_threshold[j])
                  {
                    local_buckets[j].push_back(*v);
                    break;
                  }
                }
              }
            }
          }
          // }
          // }
        }
      }
    }

    else
    {
//Use  out degee for bfs
#pragma omp parallel for reduction(max \
                                   : max_degree, start_node)
      for (NodeID_ i = 0; i < g.num_nodes(); i++)
      {
        if (g.out_degree(i) > max_degree)
        {
          max_degree = g.out_degree(i);
          start_node = i;
        }
      }

      list<NodeID_> queue1;
      queue1.push_back(start_node);
      visited[start_node] = true;
      for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
      {
        const int64_t &count = g.out_degree(start_node);
        if (count <= bucket_threshold[j])
        {
          local_buckets[j].push_back(start_node);
          break;
        }
      }
      NodeID_ s1;
      while (!queue1.empty())
      {
        s1 = queue1.front();
        // cout << s << " ";
        queue1.pop_front();
        // #pragma omp parallel
        // {
        // for (auto v = g.out_neigh(s).begin(); v != g.out_neigh(s).end(); v++)
        for (auto v = g.in_neigh(s1).begin(); v != g.in_neigh(s1).end(); v++)
        {
          // #pragma omp single nowait
          // {
          if (!visited[*v])
          {
            visited[*v] = true;
            // #pragma omp critical
            queue.push_back(*v);

            // #pragma omp parallel for schedule(static)
            for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
            {
              const int64_t &count = g.out_degree(*v);
              if (count <= bucket_threshold[j])
              {
                local_buckets[j].push_back(*v);
                break;
              }
            }
          }
        }
      }

      for (NodeID_ u = 0; u < g.num_nodes(); u++)
      {

        if (!visited[u])
        {
          list<NodeID_> queue;
          queue.push_back(u);
          visited[u] = true;
          for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
          {
            const int64_t &count = g.in_degree(u);
            if (count <= bucket_threshold[j])
            {
              local_buckets[j].push_back(u);
              break;
            }
          }
          NodeID_ s;

          while (!queue.empty())
          {
            s = queue.front();
            // cout << s << " ";
            queue.pop_front();
            // for (NodeID_ v : g.in_neigh(s))
            for (NodeID_ v : g.out_neigh(s))
            {
              if (!visited[v])
              {
                visited[v] = true;
                queue.push_back(v);

                // #pragma omp parallel for schedule(static)
                for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
                {
                  const int64_t &count = g.in_degree(v);
                  if (count <= bucket_threshold[j])
                  {
                    local_buckets[j].push_back(v);
                    break;
                  }
                }
              }
            }
          }
        }
      }
    }
    cout << "hi1";
    //here we have to apply reorder logic for buckets where we can set w parameter
    uint32_t temp_k = 0;
    uint32_t start_k[num_buckets];
    for (int32_t j = num_buckets - 1; j >= 0; j--)
    {
      start_k[j] = temp_k;
      temp_k += local_buckets[j].size();
      cout << "bucket no: " << j << " cumulative size of bucket: " << temp_k << endl;
    }

    // for (int32_t j = num_buckets - 1; j >= num_buckets - 2; j--)
    // {
    //   const vector<uint32_t> &current_bucket = local_buckets[j];
    //   int k = start_k[j];
    //   const size_t &size = current_bucket.size();
    //   for (uint32_t i = 0; i < size; i++)
    //   {
    //     new_ids[current_bucket[i]] = k++;
    //   }

    // }

    uint32_t new_start = 0;
    uint32_t last_2_bucket = local_buckets[0].size() + local_buckets[1].size();
    uint32_t no_nodeleft = g.num_nodes();
    ;
    uint32_t depthsize = 500;

    uint32_t depth_size[num_buckets];
    for (int32_t j = num_buckets - 1; j >= 0; j--)
    {
      // const vector<uint32_t> &current_bucket = local_buckets[j];
      double bucket_percentage = local_buckets[j].size() / g.num_nodes();
      if (bucket_percentage < .1)
      {
        depth_size[j] = local_buckets[j].size() / 10;
      }
      else
      {

        depth_size[j] = 500 * 8;
      }
    }

    uint32_t *bucket_offset;
    bucket_offset = new uint32_t[num_buckets];

    for (int32_t j = num_buckets - 1; j >= 0; j--)
    {
      bucket_offset[j] = 0;
    }

    // cout<<"hi";
    // while(no_nodeleft)
    // {
    //     for(int32_t j = num_buckets - 1; j >= 0; j--)
    //     {  const vector<uint32_t> &current_bucket = local_buckets[j];
    //       // if(current_bucket.empty())
    //       //   continue;
    //       for (int32_t k = 0; k<depth_size[j] && bucket_offset[j]<current_bucket.size(); k++)
    //       {
    //       //   if(!current_bucket.empty())
    //       // {
    //         new_ids[current_bucket[bucket_offset[j]]] = new_start++;
    //         no_nodeleft=no_nodeleft-1;

    //         bucket_offset[j]=bucket_offset[j]+1;

    //       }

    //     }
    // }

    while (no_nodeleft - last_2_bucket)
    {
      for (int32_t j = num_buckets - 1; j >= 2; j--)
      {
        const vector<uint32_t> &current_bucket = local_buckets[j];
        // if(current_bucket.empty())
        //   continue;
        for (int32_t k = 0; k < depth_size[j] && bucket_offset[j] < current_bucket.size(); k++)
        {
          //   if(!current_bucket.empty())
          // {
          new_ids[current_bucket[bucket_offset[j]]] = new_start++;
          no_nodeleft = no_nodeleft - 1;

          bucket_offset[j] = bucket_offset[j] + 1;
        }
      }
    }

    for (int32_t j = 1; j >= 0; j--)
    {
      const vector<uint32_t> &current_bucket = local_buckets[j];
      // if(current_bucket.empty())
      //   continue;
      for (int32_t k = 0; k < current_bucket.size(); k++)
      {
        //   if(!current_bucket.empty())
        // {
        new_ids[current_bucket[k]] = new_start++;
        // no_nodeleft=no_nodeleft-1;

        // bucket_offset[j]=bucket_offset[j]+1;
      }
    }

    // uint32_t last_2_bucket=local_buckets[0].size()+local_buckets[1].size();

    for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
    {
      local_buckets[j].clear();
    }

    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      if (new_ids[u] == -1)
      {
        cout << "Node id not assigned to: " << u << endl;
      }
    }

#pragma omp parallel for
    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      out_degrees[new_ids[u]] = g.out_degree(u);
    }

#pragma omp parallel for
    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      in_degrees[new_ids[u]] = g.in_degree(u);
    }

    pvector<SGOffset> out_offsets = ParallelPrefixSum(out_degrees);
    DestID_ *out_neighs = new DestID_[out_offsets[g.num_nodes()]];
    DestID_ **out_index = CSRGraph<NodeID_, DestID_>::GenIndex(out_offsets, out_neighs);
#pragma omp parallel for
    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      for (NodeID_ v : g.out_neigh(u))
        out_neighs[out_offsets[new_ids[u]]++] = new_ids[v];
      std::sort(out_index[new_ids[u]], out_index[new_ids[u] + 1]);
    }

    pvector<SGOffset> in_offsets = ParallelPrefixSum(in_degrees);
    DestID_ *in_neighs = new DestID_[in_offsets[g.num_nodes()]];
    DestID_ **in_index = CSRGraph<NodeID_, DestID_>::GenIndex(in_offsets, in_neighs);
    if (!isWeighted)
    {
#pragma omp parallel for
      for (NodeID_ u = 0; u < g.num_nodes(); u++)
      {
        for (NodeID_ v : g.in_neigh(u))
          in_neighs[in_offsets[new_ids[u]]++] = new_ids[v];
        std::sort(in_index[new_ids[u]], in_index[new_ids[u] + 1]);
      }
    }
    else
    {
#pragma omp parallel for
      for (NodeID_ u = 0; u < g.num_nodes(); u++)
      {
        for (WNode wn : g.out_neigh(u))
        {
          WNode new_node(new_ids[wn.v], wn.w);
          out_neighs[out_offsets[new_ids[u]]++] = new_node;
        }

        std::sort(out_index[new_ids[u]], out_index[new_ids[u] + 1]);
      }
    }

    t.Stop();
    PrintTime("Neighbourhood Map Time", t.Seconds());

    return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), out_index, out_neighs, in_index, in_neighs);
  }

  // For Undireected Neighbour

  static CSRGraph<NodeID_, DestID_, invert> undirectedgenerateNumberNodeNeighbourMapping(
      const CSRGraph<NodeID_, DestID_, invert> &g, bool useOutdeg, pvector<NodeID_> &new_ids, bool isWeighted = false)
  {
    typedef int32_t WeightT;
    typedef NodeWeight<NodeID_, WeightT> WNode;
    // typedef CSRGraph<NodeID_, WNode> WGraph;
    if (g.directed())
    {
      std::cout << "Cannot relabel directed graph" << std::endl;
      std::exit(-11);
    }
    Timer t;
    t.Start();
    cout << "Hi" << endl;
    auto num_vertices = g.num_nodes();
    auto num_edges = g.num_edges();
    // vertex *origG    = GA.V;

    uint32_t avg_vertex = num_edges / num_vertices;
    const uint32_t &av = avg_vertex;

    uint32_t bucket_threshold[] = {av / 2, av, av * 2, av * 4, av * 8, av * 16, av * 32, av * 64, av * 128, av * 256, av * 512, static_cast<uint32_t>(-1)};
    // making sure we use 8 buckets now
    int num_buckets = 8;
    if (num_buckets > 11)
    {
      // if you really want to increase the bucket count, add more thresholds to the bucket_threshold above.
      std::cout << "Unsupported bucket size: " << num_buckets << std::endl;
      assert(0);
    }
    bucket_threshold[num_buckets - 1] = static_cast<uint32_t>(-1);
    // pvector<NodeID_> new_ids(g.num_nodes(), -1);

    vector<uint32_t> bucket_vertices[num_buckets];
    // const int num_threads = omp_get_max_threads();
    vector<uint32_t> local_buckets[num_buckets];

    pvector<NodeID_> out_degrees(g.num_nodes());

    list<NodeID_> queue;
    bool *visited = new bool[g.num_nodes()];
#pragma omp parallel for
    for (int i = 0; i < g.num_nodes(); i++)
      visited[i] = false;
    // NodeID_ start_vertex=5;
    // queue.push_back(start_vertex);
    // visited[start_vertex]=true;

    // NodeID_ s=start_vertex;
    NodeID_ start_node = 0;
    int64_t max_degree = 0;

    cout << "starting traversal";
    if (useOutdeg)
    {

      //start from max in degree

#pragma omp parallel for reduction(max \
                                   : max_degree, start_node)
      for (NodeID_ i = 0; i < g.num_nodes(); i++)
      {
        if (g.out_degree(i) > max_degree)
        {
          max_degree = g.out_degree(i);
          start_node = i;
        }
      }

      list<NodeID_> queue1;
      queue1.push_back(start_node);
      visited[start_node] = true;
      for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
      {
        const int64_t &count = g.out_degree(start_node);
        if (count <= bucket_threshold[j])
        {
          local_buckets[j].push_back(start_node);
          break;
        }
      }
      NodeID_ s1;
      while (!queue1.empty())
      {
        s1 = queue1.front();
        // cout << s << " ";
        queue1.pop_front();
        // #pragma omp parallel
        // {
        // for (auto v = g.out_neigh(s).begin(); v != g.out_neigh(s).end(); v++)
        for (auto v = g.out_neigh(s1).begin(); v != g.out_neigh(s1).end(); v++)
        {
          // #pragma omp single nowait
          // {
          if (!visited[*v])
          {
            visited[*v] = true;
            // #pragma omp critical
            queue.push_back(*v);

            // #pragma omp parallel for schedule(static)
            for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
            {
              const int64_t &count = g.out_degree(*v);
              if (count <= bucket_threshold[j])
              {
                local_buckets[j].push_back(*v);
                break;
              }
            }
          }
        }
      }

      // This loop relies on a static scheduling
      for (NodeID_ u = 0; u < g.num_nodes(); u++)
      {
        if (!visited[u])
        {
          list<NodeID_> queue;
          queue.push_back(u);
          visited[u] = true;
          for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
          {
            const int64_t &count = g.out_degree(u);
            if (count <= bucket_threshold[j])
            {
              local_buckets[j].push_back(u);
              break;
            }
          }
          NodeID_ s;
          while (!queue.empty())
          {
            s = queue.front();
            // cout << s << " ";
            queue.pop_front();
            // #pragma omp parallel
            // {
            // for (auto v = g.out_neigh(s).begin(); v != g.out_neigh(s).end(); v++)
            for (auto v = g.out_neigh(s).begin(); v != g.out_neigh(s).end(); v++)
            {
              // #pragma omp single nowait
              // {
              if (!visited[*v])
              {
                visited[*v] = true;
                // #pragma omp critical
                queue.push_back(*v);

                // #pragma omp parallel for schedule(static)
                for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
                {
                  const int64_t &count = g.out_degree(*v);
                  if (count <= bucket_threshold[j])
                  {
                    local_buckets[j].push_back(*v);
                    break;
                  }
                }
              }
            }
          }
          // }
          // }
        }
      }
    }

    else
    {

      cout << "Should use out degree only so exit";
      exit(1);
    }

    cout << "hi1";
    //here we have to apply reorder logic for buckets where we can set w parameter
    uint32_t temp_k = 0;
    uint32_t start_k[num_buckets];
    // for (int32_t j = num_buckets - 1; j >= 0; j--)
    // {
    //   start_k[j] = temp_k;
    //   temp_k += local_buckets[j].size();
    // }

    for (int32_t j = num_buckets - 1; j >= num_buckets - 2; j--)
    {
      const vector<uint32_t> &current_bucket = local_buckets[j];
      int k = start_k[j];
      const size_t &size = current_bucket.size();
      for (uint32_t i = 0; i < size; i++)
      {
        new_ids[current_bucket[i]] = k++;
      }
    }
    uint32_t last_2_bucket = local_buckets[0].size() + local_buckets[1].size();

    uint32_t new_start = local_buckets[num_buckets - 1].size() + local_buckets[num_buckets - 2].size();
    uint32_t no_nodeleft = g.num_nodes() - local_buckets[num_buckets - 1].size() - local_buckets[num_buckets - 2].size();
    // uint32_t depthsize=500;
    uint32_t *bucket_offset;
    bucket_offset = new uint32_t[num_buckets - 2];

    for (int32_t j = num_buckets - 3; j >= 0; j--)
    {
      bucket_offset[j] = 0;
    }

    uint32_t depth_size[num_buckets];
    for (int32_t j = num_buckets - 1; j >= 0; j--)
    {
      // const vector<uint32_t> &current_bucket = local_buckets[j];
      double bucket_percentage = local_buckets[j].size() / g.num_nodes();
      if (bucket_percentage < .1)
      {
        depth_size[j] = local_buckets[j].size() / 10;
      }
      else
      {

        depth_size[j] = 500 * 8;
      }
    }

    while (no_nodeleft - last_2_bucket)
    {
      for (int32_t j = num_buckets - 3; j >= 2; j--)
      {
        const vector<uint32_t> &current_bucket = local_buckets[j];
        // if(current_bucket.empty())
        //   continue;
        for (int32_t k = 0; k < depth_size[j] && bucket_offset[j] < current_bucket.size(); k++)
        {
          //   if(!current_bucket.empty())
          // {
          new_ids[current_bucket[bucket_offset[j]]] = new_start++;
          no_nodeleft = no_nodeleft - 1;

          bucket_offset[j] = bucket_offset[j] + 1;
        }
      }
    }

    for (int32_t j = 1; j >= 0; j--)
    {
      const vector<uint32_t> &current_bucket = local_buckets[j];
      // if(current_bucket.empty())
      //   continue;
      for (int32_t k = 0; k < current_bucket.size(); k++)
      {
        //   if(!current_bucket.empty())
        // {
        new_ids[current_bucket[k]] = new_start++;
        // no_nodeleft=no_nodeleft-1;

        // bucket_offset[j]=bucket_offset[j]+1;
      }
    }
    cout << "New start" << new_start << "NUM Nodes" << g.num_nodes() << endl;

    if (new_start == g.num_nodes())
    {
      cout << "ok";
    }
    else
    {
      cout << "" << new_start << "  no" << g.num_nodes() << endl;
    }

    for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
    {
      local_buckets[j].clear();
    }

    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      if (new_ids[u] == -1)
      {
        cout << "Node id not assigned to: " << u << endl;
      }
    }

#pragma omp parallel for
    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      out_degrees[new_ids[u]] = g.out_degree(u);
    }

    // #pragma omp parallel for
    //     for (NodeID_ u = 0; u < g.num_nodes(); u++)
    //     {
    //       in_degrees[new_ids[u]] = g.in_degree(u);
    //     }

    pvector<SGOffset> out_offsets = ParallelPrefixSum(out_degrees);
    DestID_ *out_neighs = new DestID_[out_offsets[g.num_nodes()]];
    DestID_ **out_index = CSRGraph<NodeID_, DestID_>::GenIndex(out_offsets, out_neighs);
    if (!isWeighted)
    {
#pragma omp parallel for
      for (NodeID_ u = 0; u < g.num_nodes(); u++)
      {
        for (NodeID_ v : g.out_neigh(u))
          out_neighs[out_offsets[new_ids[u]]++] = new_ids[v];
        std::sort(out_index[new_ids[u]], out_index[new_ids[u] + 1]);
      }
    }
    else
    {
#pragma omp parallel for
      for (NodeID_ u = 0; u < g.num_nodes(); u++)
      {
        for (WNode wn : g.out_neigh(u))
        {
          WNode new_node(new_ids[wn.v], wn.w);
          out_neighs[out_offsets[new_ids[u]]++] = new_node;
        }

        std::sort(out_index[new_ids[u]], out_index[new_ids[u] + 1]);
      }
    }

    // pvector<SGOffset> in_offsets = ParallelPrefixSum(in_degrees);
    // DestID_ *in_neighs = new DestID_[in_offsets[g.num_nodes()]];
    // DestID_ **in_index = CSRGraph<NodeID_, DestID_>::GenIndex(in_offsets, in_neighs);
    // #pragma omp parallel for
    //     for (NodeID_ u = 0; u < g.num_nodes(); u++)
    //     {
    //       for (NodeID_ v : g.in_neigh(u))
    //         in_neighs[in_offsets[new_ids[u]]++] = new_ids[v];
    //       std::sort(in_index[new_ids[u]], in_index[new_ids[u] + 1]);
    //     }

    t.Stop();
    PrintTime("Neighbourhood Map Time", t.Seconds());

    return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), out_index, out_neighs);
  }

  //...........................................................
  //-----------Numbering Node------------------

  static CSRGraph<NodeID_, DestID_, invert> generateConnectedComponentNeighbourMapping(
      const CSRGraph<NodeID_, DestID_, invert> &g, bool useOutdeg, pvector<NodeID_> &new_ids, bool isWeighted = false)
  {
    typedef int32_t WeightT;
    typedef NodeWeight<NodeID_, WeightT> WNode;
    // typedef CSRGraph<NodeID_, WNode> WGraph;
    if (!g.directed())
    {
      std::cout << "Cannot relabel un-directed graph" << std::endl;
      std::exit(-11);
    }
    Timer t;
    t.Start();

    auto num_vertices = g.num_nodes();
    auto num_edges = g.num_edges();
    // vertex *origG    = GA.V;

    uint32_t avg_vertex = num_edges / num_vertices;
    const uint32_t &av = avg_vertex;

    uint32_t bucket_threshold[] = {av / 2, av, av * 2, av * 4, av * 8, av * 16, av * 32, av * 64, av * 128, av * 256, av * 512, static_cast<uint32_t>(-1)};
    // making sure we use 8 buckets now
    int num_buckets = 8;
    if (num_buckets > 11)
    {
      // if you really want to increase the bucket count, add more thresholds to the bucket_threshold above.
      std::cout << "Unsupported bucket size: " << num_buckets << std::endl;
      assert(0);
    }
    bucket_threshold[num_buckets - 1] = static_cast<uint32_t>(-1);
    // pvector<NodeID_> new_ids(g.num_nodes(), -1);

    vector<uint32_t> bucket_vertices[num_buckets];
    // const int num_threads = omp_get_max_threads();
    vector<uint32_t> local_buckets[num_buckets];

    pvector<NodeID_> out_degrees(g.num_nodes());
    pvector<NodeID_> in_degrees(g.num_nodes());
    uint32_t new_start = 0;
    list<NodeID_> queue;
    bool *visited = new bool[g.num_nodes()];
#pragma omp parallel for
    for (int i = 0; i < g.num_nodes(); i++)
      visited[i] = false;

    NodeID_ start_node = 0;
    int64_t max_degree = 0;
    int64_t no_of_connected_component = 1;
    // cout << "starting traversal";
    if (useOutdeg)
    {

      //start from max in degree

#pragma omp parallel for reduction(max \
                                   : max_degree, start_node)
      for (NodeID_ i = 0; i < g.num_nodes(); i++)
      {
        if (g.in_degree(i) > max_degree)
        {
          max_degree = g.in_degree(i);
          start_node = i;
        }
      }

      list<NodeID_> queue1;
      queue1.push_back(start_node);
      visited[start_node] = true;
      for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
      {
        const int64_t &count = g.out_degree(start_node);
        if (count <= bucket_threshold[j])
        {
          local_buckets[j].push_back(start_node);
          break;
        }
      }
      NodeID_ s1;
      while (!queue1.empty())
      {
        s1 = queue1.front();
        // cout << s << " ";
        queue1.pop_front();
        // #pragma omp parallel
        // {
        // for (auto v = g.out_neigh(s).begin(); v != g.out_neigh(s).end(); v++)
        for (auto v = g.in_neigh(s1).begin(); v != g.in_neigh(s1).end(); v++)
        {
          // #pragma omp single nowait
          // {
          if (!visited[*v])
          {
            visited[*v] = true;
            // #pragma omp critical
            queue.push_back(*v);

            // #pragma omp parallel for schedule(static)
            for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
            {
              const int64_t &count = g.out_degree(*v);
              if (count <= bucket_threshold[j])
              {
                local_buckets[j].push_back(*v);
                break;
              }
            }
          }
        }
      }

      for (int32_t j = num_buckets - 3; j >= 2; j--)
      {
        const vector<uint32_t> &current_bucket = local_buckets[j];

        for (int32_t k = 0; k < current_bucket.size(); k++)
        {
          //   if(!current_bucket.empty())
          // {
          new_ids[current_bucket[k]] = new_start++;
        }
        local_buckets[j].clear();
      }

      // This loop relies on a static scheduling
      for (NodeID_ u = 0; u < g.num_nodes(); u++)
      {
        if (!visited[u])
        {
          no_of_connected_component++;
          list<NodeID_> queue;
          queue.push_back(u);
          visited[u] = true;
          for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
          {
            const int64_t &count = g.out_degree(u);
            if (count <= bucket_threshold[j])
            {
              local_buckets[j].push_back(u);
              break;
            }
          }
          NodeID_ s;
          while (!queue.empty())
          {
            s = queue.front();
            // cout << s << " ";
            queue.pop_front();
            // #pragma omp parallel
            // {
            // for (auto v = g.out_neigh(s).begin(); v != g.out_neigh(s).end(); v++)
            for (auto v = g.in_neigh(s).begin(); v != g.in_neigh(s).end(); v++)
            {
              // #pragma omp single nowait
              // {
              if (!visited[*v])
              {
                visited[*v] = true;
                // #pragma omp critical
                queue.push_back(*v);

                // #pragma omp parallel for schedule(static)
                for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
                {
                  const int64_t &count = g.out_degree(*v);
                  if (count <= bucket_threshold[j])
                  {
                    local_buckets[j].push_back(*v);
                    break;
                  }
                }
              }
            }
          }
          // }
          // }
        }

        for (int32_t j = num_buckets - 3; j >= 2; j--)
        {
          const vector<uint32_t> &current_bucket = local_buckets[j];

          for (int32_t k = 0; k < current_bucket.size(); k++)
          {
            new_ids[current_bucket[k]] = new_start++;
          }
          local_buckets[j].clear();
        }
      }

      for (int32_t j = num_buckets - 1; j >= num_buckets - 2; j--)
      {
        const vector<uint32_t> &current_bucket = local_buckets[j];

        for (int32_t k = 0; k < current_bucket.size(); k++)
        {
          new_ids[current_bucket[k]] = new_start++;
        }
        local_buckets[j].clear();
      }

      for (int32_t j = 1; j >= 0; j--)
      {
        const vector<uint32_t> &current_bucket = local_buckets[j];

        for (int32_t k = 0; k < current_bucket.size(); k++)
        {
          new_ids[current_bucket[k]] = new_start++;
        }
        local_buckets[j].clear();
      }
    }

    else
    {
      //Use  out degee for bfs
#pragma omp parallel for reduction(max \
                                   : max_degree, start_node)
      for (NodeID_ i = 0; i < g.num_nodes(); i++)
      {
        if (g.out_degree(i) > max_degree)
        {
          max_degree = g.out_degree(i);
          start_node = i;
        }
      }

      list<NodeID_> queue1;
      queue1.push_back(start_node);
      visited[start_node] = true;
      for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
      {
        const int64_t &count = g.out_degree(start_node);
        if (count <= bucket_threshold[j])
        {
          local_buckets[j].push_back(start_node);
          break;
        }
      }
      NodeID_ s1;
      while (!queue1.empty())
      {
        s1 = queue1.front();
        // cout << s << " ";
        queue1.pop_front();
        // #pragma omp parallel
        // {
        // for (auto v = g.out_neigh(s).begin(); v != g.out_neigh(s).end(); v++)
        for (auto v = g.in_neigh(s1).begin(); v != g.in_neigh(s1).end(); v++)
        {
          // #pragma omp single nowait
          // {
          if (!visited[*v])
          {
            visited[*v] = true;
            // #pragma omp critical
            queue.push_back(*v);

            // #pragma omp parallel for schedule(static)
            for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
            {
              const int64_t &count = g.out_degree(*v);
              if (count <= bucket_threshold[j])
              {
                local_buckets[j].push_back(*v);
                break;
              }
            }
          }
        }
      }

      for (int32_t j = num_buckets - 3; j >= 2; j--)
      {
        const vector<uint32_t> &current_bucket = local_buckets[j];

        for (int32_t k = 0; k < current_bucket.size(); k++)
        {
          new_ids[current_bucket[k]] = new_start++;
        }
        local_buckets[j].clear();
      }

      for (NodeID_ u = 0; u < g.num_nodes(); u++)
      {

        if (!visited[u])
        {
          no_of_connected_component++;

          list<NodeID_> queue;
          queue.push_back(u);
          visited[u] = true;
          for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
          {
            const int64_t &count = g.in_degree(u);
            if (count <= bucket_threshold[j])
            {
              local_buckets[j].push_back(u);
              break;
            }
          }
          NodeID_ s;

          while (!queue.empty())
          {
            s = queue.front();
            // cout << s << " ";
            queue.pop_front();
            // for (NodeID_ v : g.in_neigh(s))
            for (NodeID_ v : g.out_neigh(s))
            {
              if (!visited[v])
              {
                visited[v] = true;
                queue.push_back(v);

                // #pragma omp parallel for schedule(static)
                for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
                {
                  const int64_t &count = g.in_degree(v);
                  if (count <= bucket_threshold[j])
                  {
                    local_buckets[j].push_back(v);
                    break;
                  }
                }
              }
            }
          }
        }

        for (int32_t j = num_buckets - 3; j >= 2; j--)
        {
          const vector<uint32_t> &current_bucket = local_buckets[j];

          for (int32_t k = 0; k < current_bucket.size(); k++)
          {
            new_ids[current_bucket[k]] = new_start++;
          }
          local_buckets[j].clear();
        }
      }

      for (int32_t j = num_buckets - 1; j >= num_buckets - 2; j--)
      {
        const vector<uint32_t> &current_bucket = local_buckets[j];

        for (int32_t k = 0; k < current_bucket.size(); k++)
        {
          new_ids[current_bucket[k]] = new_start++;
        }
        local_buckets[j].clear();
      }

      for (int32_t j = 1; j >= 0; j--)
      {
        const vector<uint32_t> &current_bucket = local_buckets[j];

        for (int32_t k = 0; k < current_bucket.size(); k++)
        {
          new_ids[current_bucket[k]] = new_start++;
        }
        local_buckets[j].clear();
      }
    }

    // for (int32_t j = num_buckets - 1; j >= 0; j--)
    // {

    //   cout<<"Bucket size of "<<j<<"is= "<<local_buckets[j].size();
    // }

    // cout<<"Number of no_of_connected_component "<<no_of_connected_component<<endl;

    for (unsigned int j = 0; j < (unsigned)num_buckets; j++)
    {
      local_buckets[j].clear();
    }

    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      if (new_ids[u] == -1)
      {
        cout << "Node id not assigned to: " << u << endl;
      }
    }

#pragma omp parallel for
    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      out_degrees[new_ids[u]] = g.out_degree(u);
    }

#pragma omp parallel for
    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      in_degrees[new_ids[u]] = g.in_degree(u);
    }

    pvector<SGOffset> out_offsets = ParallelPrefixSum(out_degrees);
    DestID_ *out_neighs = new DestID_[out_offsets[g.num_nodes()]];
    DestID_ **out_index = CSRGraph<NodeID_, DestID_>::GenIndex(out_offsets, out_neighs);

    if (!isWeighted)
    {
#pragma omp parallel for
      for (NodeID_ u = 0; u < g.num_nodes(); u++)
      {
        for (NodeID_ v : g.out_neigh(u))
          out_neighs[out_offsets[new_ids[u]]++] = new_ids[v];
        std::sort(out_index[new_ids[u]], out_index[new_ids[u] + 1]);
      }
    }
    else
    {
#pragma omp parallel for
      for (NodeID_ u = 0; u < g.num_nodes(); u++)
      {
        for (WNode wn : g.out_neigh(u))
        {
          WNode new_node(new_ids[wn.v], wn.w);
          out_neighs[out_offsets[new_ids[u]]++] = new_node;
        }

        std::sort(out_index[new_ids[u]], out_index[new_ids[u] + 1]);
      }
    }

    pvector<SGOffset> in_offsets = ParallelPrefixSum(in_degrees);
    DestID_ *in_neighs = new DestID_[in_offsets[g.num_nodes()]];
    DestID_ **in_index = CSRGraph<NodeID_, DestID_>::GenIndex(in_offsets, in_neighs);
#pragma omp parallel for
    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      for (NodeID_ v : g.in_neigh(u))
        in_neighs[in_offsets[new_ids[u]]++] = new_ids[v];
      std::sort(in_index[new_ids[u]], in_index[new_ids[u] + 1]);
    }

    t.Stop();
    PrintTime("Connected Component Neighbourhood Map Time", t.Seconds());

    return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), out_index, out_neighs, in_index, in_neighs);
  }

  //........................ using hybrid bfs for reordering ......................
//   pvector<NodeID_> InitParent2(const CSRGraph<NodeID_, DestID_, invert> &g, bool useOutDeg)
//   {
//     pvector<NodeID_> parent(g.num_nodes());
// #pragma omp parallel for
//     for (NodeID_ n = 0; n < g.num_nodes(); n++)
//     {
//       if (useOutDeg)
//       {
//         parent[n] = g.in_degree(n) != 0 ? -g.in_degree(n) : 0;
//       }
//       else
//       {
//         parent[n] = g.out_degree(n) != 0 ? -g.out_degree(n) : 0;
//       }
//     }
//     return parent;
//   }

static void printLocalBuckets(vector<uint32_t>  (local_buckets)[NO_BUCKETS])
{
  // int num_buckets = 8;
  for(int i = 0; i < NO_BUCKETS; i++)
  {
    vector<uint32_t> curr_bucket = local_buckets[i];
    for(int j = 0; j < curr_bucket.size(); j++)
    {
      cout << curr_bucket[i] << "\t";
    }
    cout << endl;
  }
  cout << "printing local buckets complete\n";
}

static void pushToBucket(vector<uint32_t>  (local_buckets)[NO_BUCKETS], uint32_t av, NodeID_ node_num, const int64_t &count)
{
  // printLocalBuckets(local_buckets);
  // int num_buckets = 8;
  uint32_t bucket_threshold[] = {av / 2, av, av * 2, av * 4, av * 8, av * 16, av * 32, av * 64, av * 128, av * 256, av * 512, static_cast<uint32_t>(-1)};
  bucket_threshold[NO_BUCKETS - 1] = static_cast<uint32_t>(-1);

  // cout << "Pushing node_num: " << node_num << endl;
  
  for (unsigned int j = 0; j < (unsigned)NO_BUCKETS; j++)
  {
    if (count <= bucket_threshold[j])
    {
      // cout << "node_num: " << node_num << endl;
      // cout << "size of local buckets " << local_buckets[j].size() << endl;
      local_buckets[j].push_back(node_num);
      break;
    }
  }
}

static int64_t BUStep(const CSRGraph<NodeID_, DestID_, invert> &g, pvector<NodeID_> &parent, Bitmap &front,
               Bitmap &next, vector<uint32_t>  (local_buckets)[NO_THREADS_BFS][NO_BUCKETS], uint32_t av, bool useOutDeg = false) {
  int64_t awake_count = 0;
  next.reset();
  if(!useOutDeg){
    #pragma omp parallel for reduction(+ : awake_count) schedule(dynamic, 1024)
    for (NodeID_ u=0; u < g.num_nodes(); u++) {
      if (parent[u] < 0) {
        for (NodeID_ v : g.in_neigh(u)) {
          if (front.get_bit(v)) {
            parent[u] = v;
            
            // cout << "[BUStep] parent[" << u << "] = " << v << endl;
            const int64_t &count =  g.in_degree(u) ;

            // #pragma omp critical
            // {
            
            pushToBucket(local_buckets[omp_get_thread_num()], av, u, count); //put it inside critical section
            // }
            awake_count++;
            next.set_bit(u);
            break;
          }
        }
      }
    }
  }
  else{
    #pragma omp parallel for reduction(+ : awake_count) schedule(dynamic, 1024)
    for (NodeID_ u=0; u < g.num_nodes(); u++) {
      if (parent[u] < 0) {
        for (NodeID_ v : g.out_neigh(u)) {
          if (front.get_bit(v)) {
            parent[u] = v;
            // cout << "[BUStep] parent[" << u << "] = " << v << endl;
            const int64_t &count =  g.out_degree(u);

            // #pragma omp critical
            // {
              pushToBucket(local_buckets[omp_get_thread_num()], av, u, count);  //put it inside critical sections
            // }
            awake_count++;
            next.set_bit(u);
            break;
          }
        }
      }
    }
  }
  return awake_count;
}


static int64_t TDStep(const CSRGraph<NodeID_, DestID_, invert> &g, pvector<NodeID_> &parent,
               SlidingQueue<NodeID_> &queue, vector<uint32_t>  (local_buckets)[NO_THREADS_BFS][NO_BUCKETS], uint32_t av, bool useOutDeg = false) {
  int64_t scout_count = 0;
  if(!useOutDeg)
  {
    #pragma omp parallel
    {
      QueueBuffer<NodeID_> lqueue(queue);
      #pragma omp for reduction(+ : scout_count)
      for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
        NodeID_ u = *q_iter;
        for (NodeID_ v : g.out_neigh(u)) {
          NodeID_ curr_val = parent[v];
          if (curr_val < 0) {
            if (compare_and_swap(parent[v], curr_val, u)) {
              lqueue.push_back(v);

              // cout << "[TDStep] parent[" << v << "] = " << u << endl;
              const int64_t &count = g.in_degree(v) ;

              // #pragma omp critical
              // {
                pushToBucket(local_buckets[omp_get_thread_num()], av, v, count);  //put it inside critical sections
              // }
              scout_count += -curr_val;
            }
          }
        }
      }
      lqueue.flush();
    }
  }
  else{
    #pragma omp parallel
    {
      QueueBuffer<NodeID_> lqueue(queue);
      #pragma omp for reduction(+ : scout_count)
      for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
        NodeID_ u = *q_iter;
        // cout << "[TDStep] u: " << u << endl;
        for (NodeID_ v : g.in_neigh(u)) {
          NodeID_ curr_val = parent[v];
          if (curr_val < 0) {
            if (compare_and_swap(parent[v], curr_val, u)) {

              const int64_t &count =  g.out_degree(v);

              // cout << "[TDStep] parent[" << v << "] = " << u << endl;

              // #pragma omp critical
              // {
                pushToBucket(local_buckets[omp_get_thread_num()], av, v, count);  //put it inside critical sections
              // }
              lqueue.push_back(v);
              scout_count += -curr_val;
            }
          }
        }
      }
      lqueue.flush();
    }
    // cout << "[TDStep]: scout_count: " << scout_count << endl;
  }
  return scout_count;
}


static void QueueToBitmap(const SlidingQueue<NodeID_> &queue, Bitmap &bm) {
  #pragma omp parallel for
  for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
    NodeID_ u = *q_iter;
    bm.set_bit_atomic(u);
  }
}

static void BitmapToQueue(const CSRGraph<NodeID_, DestID_, invert> &g, const Bitmap &bm,
                   SlidingQueue<NodeID_> &queue) {
  #pragma omp parallel
  {
    QueueBuffer<NodeID_> lqueue(queue);
    #pragma omp for
    for (NodeID_ n=0; n < g.num_nodes(); n++)
      if (bm.get_bit(n))
        lqueue.push_back(n);
    lqueue.flush();
  }
  queue.slide_window();
}

static void DOBFS(const CSRGraph<NodeID_, DestID_, invert> &g, NodeID_ source, vector<uint32_t>  (local_buckets)[NO_THREADS_BFS][NO_BUCKETS], pvector<NodeID_> &parent, uint32_t av, bool useOutDeg = false, int alpha = 15,
                      int beta = 18) {

  // PrintStep("Source", static_cast<int64_t>(source));

  Timer t;
  // cout << "DOBFS started from source = " << source << endl;

  //Pushing source vertex to the bucket
  const int64_t &count = useOutDeg ? g.out_degree(source): g.in_degree(source);
  pushToBucket(local_buckets[omp_get_thread_num()], av, source, count);
  
  //changing source
  parent[source] = source;

  // cout << "[DOBFS] parent[" << source << "] = " << source << endl;

  SlidingQueue<NodeID_> queue(g.num_nodes());
  queue.push_back(source);
  queue.slide_window();
  Bitmap curr(g.num_nodes());
  curr.reset();
  Bitmap front(g.num_nodes());
  front.reset();
  int64_t edges_to_check = g.num_edges_directed();
  int64_t scout_count = !useOutDeg ? g.out_degree(source): g.in_degree(source);
  while (!queue.empty()) {
    if (scout_count > edges_to_check / alpha) {
      int64_t awake_count, old_awake_count;
      TIME_OP(t, QueueToBitmap(queue, front));
      PrintStep("e", t.Seconds());
      awake_count = queue.size();
      queue.slide_window();
      do {
        t.Start();
        old_awake_count = awake_count;
        awake_count = BUStep(g, parent, front, curr, local_buckets, av, useOutDeg);
        front.swap(curr);
        t.Stop();
        // PrintStep("bu", t.Seconds(), awake_count);
      } while ((awake_count >= old_awake_count) ||
               (awake_count > g.num_nodes() / beta));
      TIME_OP(t, BitmapToQueue(g, front, queue));
      PrintStep("c", t.Seconds());
      scout_count = 1;
    } else {
      t.Start();
      edges_to_check -= scout_count;
      scout_count = TDStep(g, parent, queue, local_buckets, av, useOutDeg);
      queue.slide_window();
      t.Stop();
      // PrintStep("td", t.Seconds(), queue.size());
    }
  }
  // #pragma omp parallel for
  // for (NodeID n = 0; n < g.num_nodes(); n++)
  //   if (parent[n] < -1)
  //     parent[n] = -1;

}

//Adding function to compare the node's degree and NodeID_
typedef pair<int64_t, NodeID_> deg_node_pair;

deg_node_pair myMax(deg_node_pair a, deg_node_pair b)
{
  return a.first > b.first ? a : b;
}

static CSRGraph<NodeID_, DestID_, invert> generateNeighbourMapping_ParallelBFS(
      const CSRGraph<NodeID_, DestID_, invert> &g, bool useOutdeg, pvector<NodeID_> &new_ids, bool need_weights = false, bool isWeighted = false)
  {
    omp_set_num_threads(NO_THREADS_BFS);
    typedef int32_t WeightT;
    typedef NodeWeight<NodeID_, WeightT> WNode;
    // typedef CSRGraph<NodeID_, WNode> WGraph;
    if (!g.directed())
    {
      std::cout << "Cannot relabel un-directed graph" << std::endl;
      std::exit(-11);
    }
    Timer t;
    t.Start();

    auto num_vertices = g.num_nodes();
    auto num_edges = g.num_edges();

    uint32_t avg_vertex = num_edges / num_vertices;
    const uint32_t &av = avg_vertex;

    // uint32_t bucket_threshold[] = {av / 2, av, av * 2, av * 4, av * 8, av * 16, av * 32, av * 64, av * 128, av * 256, av * 512, static_cast<uint32_t>(-1)};
    // // making sure we use 8 buckets now
    int num_buckets = 8;
    if (num_buckets > 11)
    {
      // if you really want to increase the bucket count, add more thresholds to the bucket_threshold above.
      std::cout << "Unsupported bucket size: " << num_buckets << std::endl;
      assert(0);
    }
    // bucket_threshold[num_buckets - 1] = static_cast<uint32_t>(-1);

    vector<uint32_t> bucket_vertices[NUM_BUCKETS];
    vector<uint32_t> local_buckets[NO_THREADS_BFS][NUM_BUCKETS];

    pvector<NodeID_> out_degrees(g.num_nodes());
    pvector<NodeID_> in_degrees(g.num_nodes());

    list<NodeID_> queue;
    
    t.Start();
    
    // pvector<NodeID_> parent = InitParent2(g, useOutdeg);

    pvector<NodeID_> parent(g.num_nodes());
#pragma omp parallel for
    for (NodeID_ n = 0; n < g.num_nodes(); n++)
    {
      if (useOutdeg)
      {
        parent[n] = g.in_degree(n) != 0 ? -g.in_degree(n) : -1;
      }
      else
      {
        parent[n] = g.out_degree(n) != 0 ? -g.out_degree(n) : -1;
      }
    }

    t.Stop();
    PrintStep("i", t.Seconds());

    NodeID_ start_node = 0;
    int64_t max_degree = 0;

    NodeID_ starting_nodes[NO_THREADS_BFS]={0};
    
    int64_t max_degrees[NO_THREADS_BFS] = {0};

    if (useOutdeg)
    {
//start from max in degree
      #pragma omp parallel for
      for (NodeID_ i = 0; i < g.num_nodes(); i++)
      {
        if (g.in_degree(i) > max_degrees[omp_get_thread_num()])
        {
          max_degrees[omp_get_thread_num()] = g.in_degree(i);
          starting_nodes[omp_get_thread_num()] = i;
        }
      }
    }

    else
    {
//Use  out degee for bfs
      #pragma omp parallel for
      for (NodeID_ i = 0; i < g.num_nodes(); i++)
      {
        if (g.out_degree(i) > max_degrees[omp_get_thread_num()])
        {
          max_degrees[omp_get_thread_num()] = g.out_degree(i);
          starting_nodes[omp_get_thread_num()] = i;
        }
      }
    }

    for(int i = 0; i < NO_THREADS_BFS; i++)
    {
      if(max_degree < max_degrees[i])
      {
        max_degree = max_degrees[i];
        start_node = starting_nodes[i];
      }
    }

    cout << "First start_node: " << start_node << endl;

    DOBFS(g, start_node, local_buckets, parent, av, useOutdeg);
    for(NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      if(parent[u] < 0) 
      {//unvisited node
        DOBFS(g, u, local_buckets, parent, av, useOutdeg);
      }
    }

    //here we have to apply reorder logic for buckets where we can set w parameter
    int temp_k = 0;
    uint32_t start_k[NO_THREADS_BFS][NO_BUCKETS];

    for (int32_t j = num_buckets - 1; j >= 0; j--)
    {
      for(int t = 0; t < NO_THREADS_BFS; t++){
        start_k[t][j] = temp_k;
        temp_k += local_buckets[t][j].size();
      }
    }

    cout << "temp_k: " << temp_k << endl;

    #pragma omp parallel for 
    for (int t = 0; t < NO_THREADS_BFS; t++)
    {
      for (int32_t j = num_buckets - 1; j >= 0; j--)
      {
        const vector<uint32_t> &current_bucket = local_buckets[t][j];
        int k = start_k[t][j];
        const size_t &size = current_bucket.size();
        for (uint32_t i = 0; i < size; i++)
        {
          new_ids[current_bucket[i]] = k++;
        }
      }
    }

    for(int t = 0; t < NO_THREADS_BFS; t++)
    {
      for (unsigned int j = 0; j < (unsigned)NUM_BUCKETS; j++)
      {
        local_buckets[t][j].clear();
      }
    }
#pragma omp parallel for
    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      out_degrees[new_ids[u]] = g.out_degree(u);
    }

#pragma omp parallel for
    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      in_degrees[new_ids[u]] = g.in_degree(u);
    }
    pvector<SGOffset> out_offsets = ParallelPrefixSum(out_degrees);
    DestID_ *out_neighs = new DestID_[out_offsets[g.num_nodes()]];
    DestID_ **out_index = CSRGraph<NodeID_, DestID_>::GenIndex(out_offsets, out_neighs);
    if (!isWeighted)
    {
#pragma omp parallel for
      for (NodeID_ u = 0; u < g.num_nodes(); u++)
      {
        for (NodeID_ v : g.out_neigh(u))
          out_neighs[out_offsets[new_ids[u]]++] = new_ids[v];
        std::sort(out_index[new_ids[u]], out_index[new_ids[u] + 1]);
      }
    }
    else
    {
#pragma omp parallel for
      for (NodeID_ u = 0; u < g.num_nodes(); u++)
      {
        for (WNode wn : g.out_neigh(u))
        {
          WNode new_node(new_ids[wn.v], wn.w);
          out_neighs[out_offsets[new_ids[u]]++] = new_node;
        }

        std::sort(out_index[new_ids[u]], out_index[new_ids[u] + 1]);
      }
    }

    pvector<SGOffset> in_offsets = ParallelPrefixSum(in_degrees);
    DestID_ *in_neighs = new DestID_[in_offsets[g.num_nodes()]];
    DestID_ **in_index = CSRGraph<NodeID_, DestID_>::GenIndex(in_offsets, in_neighs);
#pragma omp parallel for
    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      for (NodeID_ v : g.in_neigh(u))
        in_neighs[in_offsets[new_ids[u]]++] = new_ids[v];
      std::sort(in_index[new_ids[u]], in_index[new_ids[u] + 1]);
    }

    t.Stop();
    PrintTime("Neighbourhood Map Time", t.Seconds());

    return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), out_index, out_neighs, in_index, in_neighs);
  }



  static void Fusion(const CSRGraph<NodeID_, DestID_, invert> &g, list<NodeID_> &hypernode, NodeID_ v, bool* visited, int &lambda){
    list<NodeID_> queue;
    hypernode.push_back(v);
    queue.push_back(v);
    
    int hop = 2;

    while(hop>0){
      int temp_size = queue.size();

      for(int i=0; i<temp_size; i++){
        NodeID_ s = queue.front();
        queue.pop_front();

        for (auto v = g.in_neigh(s).begin(); v != g.in_neigh(s).end(); v++){
          
          if(!visited[*v] && g.out_degree(*v) < lambda){
            queue.push_back(*v);
            hypernode.push_back(*v);
          }
        }
      }

      hop--;
    }
  }



  static CSRGraph<NodeID_, DestID_, invert> generateSOrdering(
      const CSRGraph<NodeID_, DestID_, invert> &g, bool useOutdeg, pvector<NodeID_> &new_ids, bool need_weights = false, bool isWeighted = false){
    typedef int32_t WeightT;
    typedef NodeWeight<NodeID_, WeightT> WNode;
    
    if (!g.directed())
    {
      std::cout << "Cannot relabel un-directed graph" << std::endl;
      std::exit(-11);
    }
    Timer t;
    t.Start();

    // Degrees for the final graph
    pvector<NodeID_> out_degrees(g.num_nodes());
    pvector<NodeID_> in_degrees(g.num_nodes());


    // Defining variables required for SOrder

    // Visited array defining
    bool *visited = new bool[g.num_nodes()];
    //#pragma omp parallel for
    for (int i = 0; i < g.num_nodes(); i++)
      visited[i] = false;

    NodeID_ move_id = 0;
    NodeID_ seed = -1;

    int lambda = 50;

    for (NodeID_ i = 0; i < g.num_nodes(); i++){
      if(!visited[i]){
        seed = i;
        //cout<<"seed: "<<seed<<endl;
        while(seed != -1){
          list<NodeID_> hypernode;
          Fusion(g, hypernode, seed, visited, lambda); 

          list<NodeID_> hypernode_in;
          
          for(auto j = hypernode.begin(); j != hypernode.end(); j++)
            if(!visited[*j]){              
              new_ids[*j] = move_id++;
              visited[*j] = true;
            }            
          


          for(auto j = hypernode.begin(); j != hypernode.end(); j++)
            for (auto v = g.in_neigh(*j).begin(); v != g.in_neigh(*j).end(); v++)
              if(!visited[*v]){
                hypernode_in.push_back(*v);

              }
          


          seed = -1;

          list<NodeID_> NonHubs;

          while(!hypernode_in.empty()){
            NodeID_ u = hypernode_in.front();
            hypernode_in.pop_front();

            if(!visited[u]){
              //cout<<"If taken"<<endl;
              seed = u;
              if(g.out_degree(u)>= lambda){
                new_ids[u] = move_id++;
                //cout<<"new_ids["<< u<<"] = "<< new_ids[u] <<endl;
                visited[u] =true;
              }
              else
                NonHubs.push_back(u);
            }
            //else cout<<"If not taken"<<endl;

          }

          while(!NonHubs.empty()){

            NodeID_ v = NonHubs.front();
            NonHubs.pop_front();
            if(!visited[v]){
            new_ids[v] = move_id++;
            //cout<<"new_ids["<< v<<"] = "<< new_ids[v] <<endl;
            visited[v] = true;}
          }
          //cout<<"Seed = "<<seed<<endl;
        } // end of while loop
      } // end of if statement
      //else cout<<"Skipped seed = "<<i<<endl;
    } // end of for loop
cout<<"Num of nodes: "<<g.num_nodes()<<", Num of new nodes: "<<move_id<<endl;
cout<<"Completed Sordering"<<endl;
  #pragma omp parallel for
    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      out_degrees[new_ids[u]] = g.out_degree(u);
    }

#pragma omp parallel for
    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      in_degrees[new_ids[u]] = g.in_degree(u);
    }


    pvector<SGOffset> out_offsets = ParallelPrefixSum(out_degrees);
    DestID_ *out_neighs = new DestID_[out_offsets[g.num_nodes()]];
    DestID_ **out_index = CSRGraph<NodeID_, DestID_>::GenIndex(out_offsets, out_neighs);
    if (!isWeighted)
    {
#pragma omp parallel for
      for (NodeID_ u = 0; u < g.num_nodes(); u++)
      {
        for (NodeID_ v : g.out_neigh(u))
          out_neighs[out_offsets[new_ids[u]]++] = new_ids[v];
        std::sort(out_index[new_ids[u]], out_index[new_ids[u] + 1]);
      }
    }
    else
    {
#pragma omp parallel for
      for (NodeID_ u = 0; u < g.num_nodes(); u++)
      {
        for (WNode wn : g.out_neigh(u))
        {
          WNode new_node(new_ids[wn.v], wn.w);
          out_neighs[out_offsets[new_ids[u]]++] = new_node;
        }

        std::sort(out_index[new_ids[u]], out_index[new_ids[u] + 1]);
      }
    }

    pvector<SGOffset> in_offsets = ParallelPrefixSum(in_degrees);
    DestID_ *in_neighs = new DestID_[in_offsets[g.num_nodes()]];
    DestID_ **in_index = CSRGraph<NodeID_, DestID_>::GenIndex(in_offsets, in_neighs);
#pragma omp parallel for
    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      for (NodeID_ v : g.in_neigh(u))
        in_neighs[in_offsets[new_ids[u]]++] = new_ids[v];
      std::sort(in_index[new_ids[u]], in_index[new_ids[u] + 1]);
    }

    // assert(new_ids[2721009] == 2218313);
    t.Stop();
    PrintTime("Neighbourhood Map Time", t.Seconds());

    return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), out_index, out_neighs, in_index, in_neighs);
  }







  static CSRGraph<NodeID_, DestID_, invert> generateNOrdering(
      const CSRGraph<NodeID_, DestID_, invert> &g, bool useOutdeg, pvector<NodeID_> &new_ids, bool need_weights = false, bool isWeighted = false){
    typedef int32_t WeightT;
    typedef NodeWeight<NodeID_, WeightT> WNode;
    //cout<<"starting norder"<<endl;
    if (!g.directed())
    {
      std::cout << "Cannot relabel un-directed graph" << std::endl;
      std::exit(-11);
    }
    Timer t;
    t.Start();

    // Degrees for the final graph
    pvector<NodeID_> out_degrees(g.num_nodes());
    pvector<NodeID_> in_degrees(g.num_nodes());


    // Defining variables required for SOrder

    // Visited array defining
    bool *visited = new bool[g.num_nodes()];
    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes(); i++)
      visited[i] = false;

    int64_t id_new = 0;

    vector< vector<int64_t> > to_sort;

    //cout<<"preparing to sort in descending order"<<endl;

    //#pragma omp parallel for
    for(int64_t i =0; i<g.num_nodes(); i++){
      vector<int64_t> temp;
      temp.push_back(g.out_degree(i));
      temp.push_back(i);
      to_sort.push_back(temp);
    }
    //cout<<"TO_sort array formed"<<endl;


    //auto ptr = (pair<NodeID_, NodeID_>*) to_sort;
    sort(to_sort.begin(), to_sort.end());
    //cout<<"Sorted the to_sort array"<<endl;
    //cout<<"Size of graph: "<<g.num_nodes()<<endl;
    //cout<<"Size of to_sort: "<<to_sort.size()<<endl;    
    vector<NodeID_> temp_ids;

    //#pragma omp parallel for
    for(NodeID_ i =g.num_nodes()-1; i>=0; i--){
      //pair<NodeID_, NodeID_> z = *(((pair<NodeID_, NodeID_>*) to_sort) + i);
      temp_ids.push_back(to_sort[i][1]);
    }

    //cout<<"For ke baad"<<endl;




    for (NodeID_ j = 0; j < g.num_nodes(); j++){
      NodeID_ i = temp_ids[j];
      if(!visited[i]){
        //cout<<"Seed: "<<i<<" Out neighbours: "<<g.out_degree(i)<<endl;
        list<NodeID_> queue;

        queue.push_back(i);
        new_ids[i] = id_new++;
        visited[i] = true;

        //cout<<"seed = "<<i<<" new id: "<<new_ids[i]<<endl;
        int hop = 2;

        while(hop>0){
         int temp_size = queue.size();
           for(int j=0; j<temp_size; j++){
              NodeID_ s = queue.front();
              queue.pop_front();

              for (auto v = g.in_neigh(s).begin(); v != g.in_neigh(s).end(); v++){
                if(!visited[*v]){
                  //cout<<"Adding in neigh: "<< *v <<" of seed node: "<<i<<endl;
                  queue.push_back(*v);
                  visited[*v] = true;
                  new_ids[*v] = id_new++;
                }
              }
            }

          hop--;
        }
    
      }
    }
  

      #pragma omp parallel for
    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      out_degrees[new_ids[u]] = g.out_degree(u);
    }

#pragma omp parallel for
    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      in_degrees[new_ids[u]] = g.in_degree(u);
    }


    pvector<SGOffset> out_offsets = ParallelPrefixSum(out_degrees);
    DestID_ *out_neighs = new DestID_[out_offsets[g.num_nodes()]];
    DestID_ **out_index = CSRGraph<NodeID_, DestID_>::GenIndex(out_offsets, out_neighs);
    if (!isWeighted)
    {
#pragma omp parallel for
      for (NodeID_ u = 0; u < g.num_nodes(); u++)
      {
        for (NodeID_ v : g.out_neigh(u))
          out_neighs[out_offsets[new_ids[u]]++] = new_ids[v];
        std::sort(out_index[new_ids[u]], out_index[new_ids[u] + 1]);
      }
    }
    else
    {
#pragma omp parallel for
      for (NodeID_ u = 0; u < g.num_nodes(); u++)
      {
        for (WNode wn : g.out_neigh(u))
        {
          WNode new_node(new_ids[wn.v], wn.w);
          out_neighs[out_offsets[new_ids[u]]++] = new_node;
        }

        std::sort(out_index[new_ids[u]], out_index[new_ids[u] + 1]);
      }
    }

    pvector<SGOffset> in_offsets = ParallelPrefixSum(in_degrees);
    DestID_ *in_neighs = new DestID_[in_offsets[g.num_nodes()]];
    DestID_ **in_index = CSRGraph<NodeID_, DestID_>::GenIndex(in_offsets, in_neighs);
#pragma omp parallel for
    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      for (NodeID_ v : g.in_neigh(u))
        in_neighs[in_offsets[new_ids[u]]++] = new_ids[v];
      std::sort(in_index[new_ids[u]], in_index[new_ids[u] + 1]);
    }

    // assert(new_ids[2721009] == 2218313);
    t.Stop();
    PrintTime("Neighbourhood Map Time", t.Seconds());

    return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), out_index, out_neighs, in_index, in_neighs);

  }

static void getCommHotness(const CSRGraph<NodeID_, DestID_, invert> &g, NodeID_ seed, int64_t &commHotness, bool* visited, int lambda, NodeID_* commMapping, int hop){
  list<NodeID_> queue;
    
    queue.push_back(seed);
    visited[seed] = true;
    commMapping[seed] = seed;
    
    
    if(g.out_degree(seed)> lambda) commHotness++;
    
    //int hop = radius;
    while(hop>0){
      int temp_size = queue.size();
      for(int i=0; i<temp_size; i++){
        NodeID_ s = queue.front();
        queue.pop_front();  
        for (auto v = g.in_neigh(s).begin(); v != g.in_neigh(s).end(); v++){  
          if(!visited[*v]){
            commMapping[*v] = seed;
            queue.push_back(*v);
            visited[*v] = true;
            //nodes_visited++;
            if(g.out_degree(*v) > lambda) commHotness++;
          }
        }
      }
      hop--;
    }
}




static CSRGraph<NodeID_, DestID_, invert> MyReordering(
      const CSRGraph<NodeID_, DestID_, invert> &g, bool useOutdeg, pvector<NodeID_> &new_ids, bool need_weights = false, bool isWeighted = false, int rad=4){

    typedef int32_t WeightT;
    typedef NodeWeight<NodeID_, WeightT> WNode;
    //cout<<"starting norder"<<endl;
    if (!g.directed())
    {
      std::cout << "Cannot relabel un-directed graph" << std::endl;
      std::exit(-11);
    }
    Timer t;
    t.Start();
    //cout<<"Hello ji"<<endl;
    //cout<<"rad = "<< rad<<endl;
    // Degrees for the final graph
    pvector<NodeID_> out_degrees(g.num_nodes());
    pvector<NodeID_> in_degrees(g.num_nodes());


    // Defining variables required for SOrder
    
    bool *visited = new bool[g.num_nodes()];
    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes(); i++)
      visited[i] = false;
    

    
    NodeID_ *commMapping = new NodeID_[g.num_nodes()];
    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes(); i++)
      commMapping[i] = -1;
    

    int lambda = g.num_edges()/g.num_nodes()+1;
    int radius = 4;
    //cout<<"Radius value: "<< radius<<endl;

    vector< vector<int64_t> > to_sort;
    
    for(int64_t i =0; i<g.num_nodes(); i++){
      if(!visited[i]){
        vector<int64_t> temp;
        int64_t commHotness=0;
        getCommHotness(g, i, commHotness, visited, lambda, commMapping, radius);
        temp.push_back(commHotness);
        temp.push_back(i);
        to_sort.push_back(temp);
      }
    }
    
    sort(to_sort.begin(), to_sort.end());
    

    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes(); i++)
      visited[i] = false;

    NodeID_ move_id = 0;
    //cout<<"Starting actual reordering"<<endl;
    for(NodeID_ i = to_sort.size()-1; i>=0; i--){

      NodeID_ seed = to_sort[i][1];
      visited[seed] = true;
      new_ids[seed] = move_id++;
      
      //cout<<"Seed: "<<seed<<", new id: "<<new_ids[seed]<<endl;

      list<NodeID_> queue;
      list<NodeID_> cold_nodes;
      queue.push_back(seed);
      
      int hop = radius;
      
      while(hop>0){
        int temp_size = queue.size();
        for(int i=0; i<temp_size; i++){
          NodeID_ s = queue.front();
          queue.pop_front();  
          for (auto v = g.in_neigh(s).begin(); v != g.in_neigh(s).end(); v++){ 
            if(!visited[*v] && commMapping[*v]== seed){
              queue.push_back(*v);
              visited[*v] = true;
              if(g.out_degree(*v) > lambda) new_ids[*v] = move_id++;
              else cold_nodes.push_back(*v);
            }             
          }
        }
        hop--;
      }
      while(!cold_nodes.empty()){
        NodeID_ i = cold_nodes.front();
        cold_nodes.pop_front();
        new_ids[i] = move_id++; 
      }


    }

   
      #pragma omp parallel for
    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      out_degrees[new_ids[u]] = g.out_degree(u);
    }

#pragma omp parallel for
    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      in_degrees[new_ids[u]] = g.in_degree(u);
    }


    pvector<SGOffset> out_offsets = ParallelPrefixSum(out_degrees);
    DestID_ *out_neighs = new DestID_[out_offsets[g.num_nodes()]];
    DestID_ **out_index = CSRGraph<NodeID_, DestID_>::GenIndex(out_offsets, out_neighs);
    if (!isWeighted)
    {
#pragma omp parallel for
      for (NodeID_ u = 0; u < g.num_nodes(); u++)
      {
        for (NodeID_ v : g.out_neigh(u))
          out_neighs[out_offsets[new_ids[u]]++] = new_ids[v];
        std::sort(out_index[new_ids[u]], out_index[new_ids[u] + 1]);
      }
    }
    else
    {
#pragma omp parallel for
      for (NodeID_ u = 0; u < g.num_nodes(); u++)
      {
        for (WNode wn : g.out_neigh(u))
        {
          WNode new_node(new_ids[wn.v], wn.w);
          out_neighs[out_offsets[new_ids[u]]++] = new_node;
        }

        std::sort(out_index[new_ids[u]], out_index[new_ids[u] + 1]);
      }
    }

    pvector<SGOffset> in_offsets = ParallelPrefixSum(in_degrees);
    DestID_ *in_neighs = new DestID_[in_offsets[g.num_nodes()]];
    DestID_ **in_index = CSRGraph<NodeID_, DestID_>::GenIndex(in_offsets, in_neighs);
#pragma omp parallel for
    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      for (NodeID_ v : g.in_neigh(u))
        in_neighs[in_offsets[new_ids[u]]++] = new_ids[v];
      std::sort(in_index[new_ids[u]], in_index[new_ids[u] + 1]);
    }

    // assert(new_ids[2721009] == 2218313);
    t.Stop();
    PrintTime("Neighbourhood Map Time", t.Seconds());

    return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), out_index, out_neighs, in_index, in_neighs);















}


static void getCommHotness_alt(const CSRGraph<NodeID_, DestID_, invert> &g, NodeID_ seed, int64_t &commHotness, bool* visited, int lambda, NodeID_* commMapping, int hop){
  list<NodeID_> queue;
    
    queue.push_back(seed);
    visited[seed] = true;
    commMapping[seed] = seed;
    int64_t tot_degree=0, num_nod=0;
    
    
      num_nod++;
      tot_degree+= g.out_degree(seed);
    

    
    //int hop = radius;
    while(hop>0){
      int temp_size = queue.size();
      for(int i=0; i<temp_size; i++){
        NodeID_ s = queue.front();
        queue.pop_front();  
        for (auto v = g.in_neigh(s).begin(); v != g.in_neigh(s).end(); v++){  
          if(!visited[*v]){
            commMapping[*v] = seed;
            queue.push_back(*v);
            visited[*v] = true;
            //nodes_visited++;
            
              num_nod ++;
              tot_degree+= g.out_degree(*v);
            
          }
        }
      }
      hop--;
    }

  commHotness = tot_degree/num_nod;
}

static CSRGraph<NodeID_, DestID_, invert> MyReordering_alt(
      const CSRGraph<NodeID_, DestID_, invert> &g, bool useOutdeg, pvector<NodeID_> &new_ids, bool need_weights = false, bool isWeighted = false){

    typedef int32_t WeightT;
    typedef NodeWeight<NodeID_, WeightT> WNode;
    //cout<<"starting norder"<<endl;
    if (!g.directed())
    {
      std::cout << "Cannot relabel un-directed graph" << std::endl;
      std::exit(-11);
    }
    Timer t;
    t.Start();

    // Degrees for the final graph
    pvector<NodeID_> out_degrees(g.num_nodes());
    pvector<NodeID_> in_degrees(g.num_nodes());


    // Defining variables required for SOrder
    
    bool *visited = new bool[g.num_nodes()];
    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes(); i++)
      visited[i] = false;
    

    
    NodeID_ *commMapping = new NodeID_[g.num_nodes()];
    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes(); i++)
      commMapping[i] = -1;
    

    int lambda = g.num_edges()/g.num_nodes()+1;
    const int radius = 8;


    vector< vector<int64_t> > to_sort;
    
    for(int64_t i =0; i<g.num_nodes(); i++){
      if(!visited[i]){
        vector<int64_t> temp;
        int64_t commHotness=0;
        getCommHotness_alt(g, i, commHotness, visited, lambda, commMapping, radius);
        temp.push_back(commHotness);
        temp.push_back(i);
        to_sort.push_back(temp);
      }
    }
    
    sort(to_sort.begin(), to_sort.end());
    

    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes(); i++)
      visited[i] = false;

    NodeID_ move_id = 0;
    //cout<<"Starting actual reordering"<<endl;
    for(NodeID_ i = to_sort.size()-1; i>=0; i--){

      NodeID_ seed = to_sort[i][1];
      visited[seed] = true;
      new_ids[seed] = move_id++;
      
      //cout<<"Seed: "<<seed<<", new id: "<<new_ids[seed]<<endl;

      list<NodeID_> queue;
      list<NodeID_> cold_nodes;
      queue.push_back(seed);
      
      int hop = radius;
      
      while(hop>0){
        int temp_size = queue.size();
        for(int i=0; i<temp_size; i++){
          NodeID_ s = queue.front();
          queue.pop_front();  
          for (auto v = g.in_neigh(s).begin(); v != g.in_neigh(s).end(); v++){ 
            if(!visited[*v] && commMapping[*v]== seed){
              queue.push_back(*v);
              visited[*v] = true;
              if(g.out_degree(*v) > lambda) new_ids[*v] = move_id++;
              else cold_nodes.push_back(*v);
            }             
          }
        }
        hop--;
      }
      while(!cold_nodes.empty()){
        NodeID_ i = cold_nodes.front();
        cold_nodes.pop_front();
        new_ids[i] = move_id++; 
      }


    }

   
      #pragma omp parallel for
    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      out_degrees[new_ids[u]] = g.out_degree(u);
    }

#pragma omp parallel for
    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      in_degrees[new_ids[u]] = g.in_degree(u);
    }


    pvector<SGOffset> out_offsets = ParallelPrefixSum(out_degrees);
    DestID_ *out_neighs = new DestID_[out_offsets[g.num_nodes()]];
    DestID_ **out_index = CSRGraph<NodeID_, DestID_>::GenIndex(out_offsets, out_neighs);
    if (!isWeighted)
    {
#pragma omp parallel for
      for (NodeID_ u = 0; u < g.num_nodes(); u++)
      {
        for (NodeID_ v : g.out_neigh(u))
          out_neighs[out_offsets[new_ids[u]]++] = new_ids[v];
        std::sort(out_index[new_ids[u]], out_index[new_ids[u] + 1]);
      }
    }
    else
    {
#pragma omp parallel for
      for (NodeID_ u = 0; u < g.num_nodes(); u++)
      {
        for (WNode wn : g.out_neigh(u))
        {
          WNode new_node(new_ids[wn.v], wn.w);
          out_neighs[out_offsets[new_ids[u]]++] = new_node;
        }

        std::sort(out_index[new_ids[u]], out_index[new_ids[u] + 1]);
      }
    }

    pvector<SGOffset> in_offsets = ParallelPrefixSum(in_degrees);
    DestID_ *in_neighs = new DestID_[in_offsets[g.num_nodes()]];
    DestID_ **in_index = CSRGraph<NodeID_, DestID_>::GenIndex(in_offsets, in_neighs);
#pragma omp parallel for
    for (NodeID_ u = 0; u < g.num_nodes(); u++)
    {
      for (NodeID_ v : g.in_neigh(u))
        in_neighs[in_offsets[new_ids[u]]++] = new_ids[v];
      std::sort(in_index[new_ids[u]], in_index[new_ids[u] + 1]);
    }

    // assert(new_ids[2721009] == 2218313);
    t.Stop();
    PrintTime("Neighbourhood Map Time", t.Seconds());

    return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), out_index, out_neighs, in_index, in_neighs);















}

};












#endif // BUILDER1_H_
