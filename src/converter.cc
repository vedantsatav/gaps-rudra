// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <iostream>

#include "benchmark.h"
#include "builder1.h"
#include "command_line.h"
#include "graph.h"
#include "reader.h"
#include "writer.h"

using namespace std;

int main(int argc, char* argv[]) {
  CLConvert cli(argc, argv, "converter");
  cli.ParseArgs();
  
  int flag=cli.flag_reordering();

  if (cli.out_weighted()) {
    WeightedBuilder bw(cli);
    
    WGraph wg = bw.MakeGraph();
    // WGraph wg;

    // if(flag == 0){
    //   wg = bw.MakeGraph();
    // }
    // else if(flag==1){
    //   wg = WeightedBuilder::generateNeighbourMapping(wg1, true);
    // }
    // else{
    //   wg = WeightedBuilder::generateDBGMapping(wg1, true);
    // }

    wg.PrintStats();
    WeightedWriter ww(wg);
    ww.WriteGraph(cli.out_filename(), cli.out_sg());
  } else {
    Builder b(cli);
    Graph g = b.MakeGraph();
    // Graph g;

    // if(flag==0)
    // {
    //   g = b.MakeGraph(); 
    // }
    // else if(flag==1)
    // {
    //   g = Builder::generateNeighbourMapping(g1, true);
    // }
    // else{
    //   g = Builder::generateDBGMapping(g1, true);
    // }

    g.PrintStats();
    Writer w(g);
    w.WriteGraph(cli.out_filename(), cli.out_sg());
  }
  return 0;
}
