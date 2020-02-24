library(DiagrammeR)

# Data Graph 1-1
DiagrammeR::grViz("
digraph flowchart {
      # node definitions with substituted label text
      node [fontname = Helvetica, shape = rectangle]
      tab1 [label = '@@1']
      tab2 [label = '@@2']
      tab3 [label = '@@3']
      tab4 [label = '@@4']
      tab5 [label = '@@5']
      tab6 [label = '@@6']
      tab7 [label = '@@7']
      tab8 [label = '@@8']
      tab9 [label = '@@9']

      # edge definitions with the node IDs
      tab1 -> tab2;
      tab1 -> tab3;
      tab1 -> tab4;
      tab2 -> tab5;
      tab2 -> tab6;
      tab2 -> tab7;
      tab2 -> tab8;
      tab2 -> tab9;
      }

      [1]: 'midi_file.mid'
      [2]: 'Track 1'
      [3]: 'Track 2'
      [4]: 'Track ...'
      [5]: 'Pitch Number'
      [6]: 'Start Time'
      [7]: 'End Time'
      [8]: 'Velocity'
      [9]: 'Pitch Bend'
      ")

export_graph(graph_name,
             file_name = "graph_1-1.png",
             file_type = "png")
