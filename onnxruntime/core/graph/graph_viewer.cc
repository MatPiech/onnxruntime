// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <queue>

#include "core/graph/graph_viewer.h"
#include "core/graph/indexed_sub_graph.h"

namespace onnxruntime {

bool NodeCompare::operator()(const Node* n1, const Node* n2) const {
  return n1->Index() < n2->Index();
}

#if !defined(ORT_MINIMAL_BUILD)
struct PriorityNodeCompare {
  inline bool IsHighPri(const Node* n) const {
    // local statics so we can compare std::strings in the checks
    static constexpr std::string_view shape_op("Shape");
    static constexpr std::string_view size_op("Size");

    const auto& op_type = n->OpType();
    return op_type == shape_op || op_type == size_op;
  }

  // Used for std::priority_queue
  // If return false, n1 will be output first
  // If return true, n2 will be output first
  bool operator()(const Node* n1, const Node* n2) const {
    // nodes in global high priority list will be output first
    const bool isN1HighPri = IsHighPri(n1);
    const bool isN2HighPri = IsHighPri(n2);
    if (isN1HighPri != isN2HighPri) {
      return isN2HighPri;
    }

    // nodes with lower priority value will be output first
    const auto n1_priority = n1->Priority();
    const auto n2_priority = n2->Priority();
    if (n1_priority != n2_priority) {
      return n1_priority > n2_priority;
    }

#ifdef ENABLE_TRAINING

    // Sorting factors for training scenarios.
    if (n1_priority == static_cast<int>(ExecutionPriority::DEFAULT)) {
      // If both nodes are normal, prioritize outputting the forward pass node.
      //
      // Note 1: This preference arises from producer-consumer node pairs not separated by "YieldOp".
      // The producer (forward pass, contributing to YieldOp inputs) and consumer (backward pass,
      // used for gradient computation) should output in forward order to save memory.
      //
      // Note 2: MemoryOptimizer marks nodes as forward by backtracking from YieldOp's inputs.
      // Nodes reached by this backtracking, identified through their inputs, are tagged as forward.
      //
      // The nodes of forward pass will be output first
      auto n1_attrs = n1->GetAttributes();
      auto n2_attrs = n2->GetAttributes();
      int64_t n1_is_forward = static_cast<int64_t>(n1_attrs.find(kBackwardNodeAttributeName) == n1_attrs.cend()) ||
                              (n1_attrs.at(kBackwardNodeAttributeName).i() + 1) % 2;
      int64_t n2_is_forward = static_cast<int64_t>(n2_attrs.find(kBackwardNodeAttributeName) == n2_attrs.cend()) ||
                              (n2_attrs.at(kBackwardNodeAttributeName).i() + 1) % 2;
      if (n1_is_forward != n2_is_forward) {
        return n2_is_forward > n1_is_forward;
      }
    } else if (n1_priority == static_cast<int>(ExecutionPriority::LOCAL_LOW)) {
      // If both are low priority nodes, we prefer to output nodes with bigger impact first.
      // Only recompute scenarios will set the critical path impact attribute.
      //
      // Note 1: Importance of Critical Path Impact in Topological Sorting
      // In recompute scenarios, it's crucial to identify which node to execute to unblock the
      // critical path. This ensures nodes in the critical path are executed without delay.
      // For more details, refer to MemoryOptimizer's implementation.
      //
      // Note 2: Defining Critical Path Impact
      // Critical path impact is a value set during MemoryOptimizer's operation to prioritize
      // node execution. It's calculated based on the topological order of nodes and their
      // dependencies, ensuring timely execution of critical nodes. For more details, refer
      // to MemoryOptimizer's implementation.
      //
      // Note 3: This trick is not necessarily bound to LOCAL_LOW priority nodes, but we are using it for
      // recompue in MemoryOptimizer, so we add the check there. Feel free to revisit the check if it is
      // useful for other priorities.
      //
      // The nodes of bigger impact pass will be output first
      const auto& n1_attrs = n1->GetAttributes();
      const auto& n2_attrs = n2->GetAttributes();
      int64_t n1_impact = (n1_attrs.find(kRecomputeNodeCriticalPathImpact) != n1_attrs.cend())
                              ? static_cast<int64_t>(n1_attrs.at(kRecomputeNodeCriticalPathImpact).i())
                              : -1;
      int64_t n2_impact = (n2_attrs.find(kRecomputeNodeCriticalPathImpact) != n2_attrs.cend())
                              ? static_cast<int64_t>(n2_attrs.at(kRecomputeNodeCriticalPathImpact).i())
                              : -1;
      if (n1_impact != -1 && n2_impact != -1) {
        return n2_impact > n1_impact;
      }
    }

#endif

    // otherwise, nodes with lower index will be output first
    return n1->Index() > n2->Index();
  }
};
#endif

GraphViewer::GraphViewer(const Graph& graph)
    : GraphViewer(graph, nullptr) {
}

GraphViewer::GraphViewer(const Graph& graph, const IndexedSubGraph& filter_info)
    : GraphViewer(graph, &filter_info) {
}

struct GroupNode {
  GroupNode() {
  }

  void Finalize() {
    // InlinedHashSet<const NodeArg*> intermediate_args;
    std::cout << "Finalize :" << std::endl;
    for (const Node* node : nodes) {
      for (const NodeArg* arg : node->InputDefs()) {
        if (std::find(output_args.begin(), output_args.end(), arg) == output_args.end()) {
          input_args.push_back(arg);
          std::cout << "input_args  " << arg->Name() << std::endl;
        }
      }

      for (const NodeArg* arg : node->OutputDefs()) {
        if (std::find(output_args.begin(), output_args.end(), arg) == output_args.end()) {
          output_args.push_back(arg);
          std::cout << "output_args  " << arg->Name() << std::endl;
        }
      }

      // for (auto output_edge_it = node->OutputEdgesBegin(); output_edge_it != node->OutputEdgesEnd(); ++output_edge_it) {
      //   const Node* output_node = &output_edge_it->GetNode();
      //   if (std::find(nodes.begin(), nodes.end(), output_node) == nodes.end()) {
      //     output_args.push_back(node->OutputDefs()[output_edge_it->GetSrcArgIndex()]);
      //   }
      // }
    }
  }

  InlinedVector<const Node*> nodes;
  InlinedVector<const NodeArg*> input_args;
  InlinedVector<const NodeArg*> output_args;
};

void handle_group_node(const Graph* graph,
                       const NodeArg* output_arg,
                       InlinedHashMap<const NodeArg*, GroupNode*>& output_arg_to_grouped_node,
                       std::vector<NodeIndex>& node_orders,
                       InlinedVector<NodeIndex>& topo_order,
                       InlinedHashSet<const NodeArg*>& already_ready) {
  std::cout << "handle_group_node for arg named " << output_arg->Name() << std::endl;

  ORT_ENFORCE(output_arg_to_grouped_node.find(output_arg) != output_arg_to_grouped_node.end(),
              "output_arg_to_grouped_node does not contain output_arg named ", output_arg->Name());

  for (const NodeArg* input_arg : output_arg_to_grouped_node[output_arg]->input_args) {
    std::cout << "processing input_arg: " << input_arg->Name() << std::endl;
    if (already_ready.find(input_arg) == already_ready.end()) {
      std::cout << "Need hanlde process group input arg " << input_arg->Name() << " first" << std::endl;
      handle_group_node(graph, input_arg, output_arg_to_grouped_node, node_orders, topo_order, already_ready);
      std::cout << "Finish process group input arg " << input_arg->Name() << std::endl;
    }
  }

  for (const Node* n : output_arg_to_grouped_node[output_arg]->nodes) {
    std::cout << "pengwa<<<" << n->Name() << std::endl;
    node_orders.push_back(n->Index());
    topo_order.push_back(n->Index());
    // in_degree[n->Index()] = 0;
  }

  for (const NodeArg* output_arg : output_arg_to_grouped_node[output_arg]->output_args) {
    std::cout << "update already_ready for output_arg: " << output_arg->Name() << std::endl;
    already_ready.insert(output_arg);
  }
}

GraphViewer::GraphViewer(const Graph& graph, const IndexedSubGraph* filter_info)
    : graph_{&graph},
      // we can setup the filter here if needed. filtered_node_indices_ will have been populated by the time it's used
      graph_nodes_{graph_->FilteredNodes(
          filter_info ? [this](NodeIndex idx) { return filtered_node_indices_.count(idx) == 0; }
                      : ConstGraphNodes::NodeFilterFunc(nullptr))},
      filter_info_{filter_info} {
  std::vector<const Node*> leaf_nodes;
  InlinedVector<const Node*> forward_output_nodes;
  InlinedVector<const Node*> backward_input_nodes;
  const Node* yield_node = nullptr;

#ifdef ENABLE_TRAINING
  // Keep the info of shape and size nodes and their parents so that after topological sort, we can move them
  // right after their parents. This is to make sure the shape and size nodes are executed right after their parents
  // so it's possible the input tensor memory can be released as soon as possible. This is especially important
  // for non-CPU devices or for training case where some gradient graphs use only shape/size of tensors from forward.
  InlinedHashSet<NodeIndex> shape_size_nodes;
  InlinedHashMap<NodeIndex, InlinedVector<NodeIndex>> shape_size_parents;

#endif

  for (auto& node : graph_->Nodes()) {
    // This is a leaf node (without any output node)
    if (node.OutputNodesBegin() == node.OutputNodesEnd()) {
      leaf_nodes.push_back(&node);
    }
    // This is a root node (without any input node)
    if (node.InputEdgesBegin() == node.InputEdgesEnd()) {
      root_nodes_.push_back(node.Index());
    }

#ifdef ENABLE_TRAINING
    if ((node.OpType() == "Shape" || node.OpType() == "Size") && node.InputEdgesBegin() != node.InputEdgesEnd()) {
      shape_size_nodes.insert(node.Index());
      NodeIndex parent = node.InputNodesBegin()->Index();
      if (shape_size_parents.find(parent) == shape_size_parents.end()) {
        shape_size_parents[parent] = InlinedVector<NodeIndex>{node.Index()};
      } else {
        shape_size_parents[parent].push_back(node.Index());
      }
    }

    if (node.OpType() == "YieldOp") {
      yield_node = &node;
      for (auto input_it = node.InputNodesBegin(); input_it != node.InputNodesEnd(); ++input_it) {
        forward_output_nodes.push_back(&*input_it);
      }

      for (auto output_it = node.OutputNodesBegin(); output_it != node.OutputNodesEnd(); ++output_it) {
        backward_input_nodes.push_back(&*output_it);
      }
    }
#endif
  }

  graph.ReverseDFSFrom(
      leaf_nodes,
      nullptr,
      [this](const Node* n) {
        nodes_in_topological_order_.push_back(n->Index());
      },
      NodeCompare());

#ifdef ENABLE_TRAINING
  auto original = std::move(nodes_in_topological_order_);
  nodes_in_topological_order_.reserve(original.size());
  InlinedHashSet<NodeIndex> visited;
  for (auto& node : original) {
    if (visited.find(node) != visited.end()) {
      continue;
    }
    nodes_in_topological_order_.push_back(node);
    visited.insert(node);
    if (shape_size_parents.find(node) != shape_size_parents.end()) {
      for (auto& following_node : shape_size_parents[node]) {
        nodes_in_topological_order_.push_back(following_node);
        visited.insert(following_node);
      }
    }
  }

#endif

#if !defined(ORT_MINIMAL_BUILD)
  // key is the first node consuming the branch subgraph.
  InlinedHashMap<const Node*, InlinedVector<const Node*>> branch_subgraphs;

  if (yield_node) {
    std::vector<NodeIndex> node_orders;
    InlinedHashSet<const Node*> nodes_before_yieldop;

    // Reverse DFS from forward output nodes to find all "forward" nodes.
    // The forward nodes are ordered by Reverse DFS tranverse.
    graph.ReverseDFSFrom(
        forward_output_nodes,
        nullptr,
        [&nodes_before_yieldop, &node_orders](const Node* n) {
          nodes_before_yieldop.insert(n);
          node_orders.push_back(n->Index());
        },
        NodeCompare());

    std::vector<NodeIndex> updated_node_orders = node_orders;
    // Considering the foward node + Shape/Size is enough.
    for (const NodeIndex& node_index : shape_size_nodes) {
      for (auto& parent : shape_size_parents[node_index]) {
        auto it = std::find(updated_node_orders.begin(), updated_node_orders.end(), parent);
        if (it != updated_node_orders.end()) {
          updated_node_orders.insert(it + 1, node_index);
          nodes_before_yieldop.insert(graph.GetNode(node_index));
        }
      }
    }

    node_orders = std::move(updated_node_orders);

    auto sort = [this, &nodes_before_yieldop, &backward_input_nodes, &graph](
                    const std::function<void(const Node*)>& enter,
                    const std::function<bool(const Node*, const Node*)>& comp,
                    std::vector<NodeIndex>& node_orders) {
      size_t number_of_nodes = NumberOfNodes() - node_orders.size();
      InlinedVector<size_t> in_degree(MaxNodeIndex(), 0);
      InlinedVector<NodeIndex> topo_order;
      topo_order.reserve(number_of_nodes);
      VisitorPriorityQueue<const Node*> to_visit(comp);
      // InlinedHashMap<const Node*, GroupedNode> node_to_grouped_node;
      InlinedVector<const Node*> branch_input_nodes;

      InlinedHashSet<const NodeArg*> already_ready;

      for (const NodeArg* input : GetInputsIncludingInitializers()) {
        already_ready.insert(input);
        std::cout << "update already_ready for input: " << input->Name() << std::endl;
      }

      for (auto& [name, tensor] : GetAllInitializedTensors()) {
        already_ready.insert(GetNodeArg(name));
        std::cout << "update already_ready for initializer: " << name << std::endl;
      }

      for (auto& node : Nodes()) {
        // Ignore forward.
        if (nodes_before_yieldop.find(&node) != nodes_before_yieldop.end()) {
          continue;
        }

        if (node.OpType() == "YieldOp") {
          in_degree[node.Index()] = 0;
          std::cout << "push " << node.Name() << " into queue (YieldOp)" << std::endl;
          to_visit.push(&node);
          continue;
        }

        size_t input_edge_count = node.GetInputEdgesCount();
        in_degree[node.Index()] = input_edge_count;
        // input_edge_count could be 0 if it takes graph input directly.

        if (input_edge_count == 0) {  // A shortcut.
          std::cout << "push " << node.Name() << " into branch_input_nodes" << std::endl;
          branch_input_nodes.push_back(&node);
          continue;
        }

        for (auto input_edge_it = node.InputEdgesBegin(); input_edge_it != node.InputEdgesEnd(); ++input_edge_it) {
          const Node* input_node = &input_edge_it->GetNode();
          // If the input edge connect to forward nodes, then we remove the in_degree of the node.
          if (nodes_before_yieldop.find(input_node) != nodes_before_yieldop.end()) {
            input_edge_count--;
            already_ready.insert(node.InputDefs()[input_edge_it->GetDstArgIndex()]);
          }
        }

        in_degree[node.Index()] = input_edge_count;
        if (input_edge_count == 0) {
          branch_input_nodes.push_back(&node);
          std::cout << "push " << node.Name() << " into branch_input_nodes" << std::endl;
        }
      }

      // Loop through the branch_input_nodes to find the branch subgraphs by its output edges in BFS,
      // and find the maximum self_contained subgraph taking the branch_input_nodes as input nodes.

      InlinedVector<const Node*> branch_subgraph;
      std::queue<const Node*> to_visit_queue;
      InlinedHashSet<const NodeArg*> branch_trigger_node_args;
      InlinedVector<size_t> in_degree_copy = in_degree;

      // add all nodes in branch_input_nodes to the queue
      for (auto branch_input_node : branch_input_nodes) {
        to_visit_queue.push(branch_input_node);
        branch_subgraph.push_back(branch_input_node);
      }
      // to_visit.push(branch_input_node);
      // visited.insert(branch_input_node);

      while (!to_visit_queue.empty()) {
        const Node* current = to_visit_queue.front();
        to_visit_queue.pop();

        if (!current) continue;

        // if (enter) {
        //   enter(current);
        // }

        for (auto node_it = current->OutputNodesBegin(); node_it != current->OutputNodesEnd(); ++node_it) {
          auto& node_in_degree = in_degree_copy[node_it->Index()];
          node_in_degree--;

          if (node_in_degree == 0) {
            to_visit_queue.push(&*node_it);
            branch_subgraph.push_back(&*node_it);
            for (const NodeArg* output_arg : node_it->OutputDefs()) {
              branch_trigger_node_args.insert(output_arg);
            }
          }
        }
      }

      // At this point, branch_subgraph is a bigger subgraph that contains all the nodes that are purely
      // triggered by the branch_input_nodes.
      InlinedVector<std::pair<const Node*, size_t>> branch_subgraph_consumers;
      InlinedVector<const NodeArg*> branch_subgraph_output_args;
      for (const Node* n : branch_subgraph) {
        for (auto output_it = n->OutputEdgesBegin(); output_it != n->OutputEdgesEnd(); ++output_it) {
          const Node* output_node = &output_it->GetNode();
          const size_t dest_in_port = output_it->GetDstArgIndex();
          if (std::find(branch_subgraph.begin(), branch_subgraph.end(), output_node) == branch_subgraph.end()) {
            branch_subgraph_consumers.push_back({output_node, dest_in_port});
            if (std::find(branch_subgraph_output_args.begin(), branch_subgraph_output_args.end(),
                          output_node->InputDefs()[dest_in_port]) == branch_subgraph_output_args.end()) {
              branch_subgraph_output_args.push_back(output_node->InputDefs()[dest_in_port]);
            }
          }
        }
      }

      InlinedVector<const NodeArg*> branch_subgraph_input_args;
      for (const Node* n : branch_subgraph) {
        for (const NodeArg* input_arg : n->InputDefs()) {
          if (std::find(branch_subgraph_output_args.begin(), branch_subgraph_output_args.end(), input_arg) ==
              branch_subgraph_output_args.end()) {
            branch_subgraph_input_args.push_back(input_arg);
          }
        }
      }

      GroupNode initial_group;
      initial_group.nodes = branch_subgraph;
      // Print the input and output node of the branch subgraph.
      std::cout << "Branch subgraph input nodes: ";
      for (const NodeArg* arg : initial_group.input_args) {
        std::cout << arg->Name() << " ";
      }
      std::cout << std::endl;
      std::cout << "Branch subgraph output nodes: ";
      for (const NodeArg* arg : initial_group.output_args) {
        std::cout << arg->Name() << " ";
      }
      std::cout << std::endl;

      // Reverse DFS from initial graph outputs (e.g. branch_subgraph_consumers) to tag each nodes:
      // If one node N contributes to a graph output A, then we will tag A to N.
      // If the node N contributes to multiple graph outputs A, B, C, then we will tag the A, B, C to N.
      InlinedHashMap<const Node*, std::set<const NodeArg*>> node_to_its_associated_outputs;
      for (const auto& consumer : branch_subgraph_consumers) {
        const NodeArg* output_arg = consumer.first->InputDefs()[consumer.second];
        const Node* end_node = GetProducerNode(output_arg->Name());
        InlinedVector<const Node*> end_nodes{end_node};
        graph.ReverseDFSFrom(
            end_nodes,
            nullptr,
            [&node_to_its_associated_outputs, &output_arg](const Node* n) {
              std::cout << "Node named " << n->Name() << " is associated with output " << output_arg->Name() << std::endl;
              node_to_its_associated_outputs[n].insert(output_arg);
            },
            nullptr,
            [&nodes_before_yieldop](const Node*, const Node* to) -> bool {
              if (nodes_before_yieldop.find(to) != nodes_before_yieldop.end()) {
                return true;  // Skip forward nodes.
              }

              if (to->OpType() == "YieldOp") {
                return true;  // Skip YieldOp. In theory, we should not reach here.
              }

              return false;
            });
      }

      // Cluster the nodes in the branch_subgraph based on the associated outputs.
      InlinedHashMap<std::set<const NodeArg*>, GroupNode> output_to_grouped_node;

      for (const auto& node : branch_subgraph) {
        const auto& associated_outputs = node_to_its_associated_outputs[node];

        if (output_to_grouped_node.find(associated_outputs) == output_to_grouped_node.end()) {
          output_to_grouped_node[associated_outputs] = GroupNode();
        }

        output_to_grouped_node[associated_outputs].nodes.push_back(node);
      }

      for (auto& [output_args, grouped_node] : output_to_grouped_node) {
        grouped_node.Finalize();
      }

      // Flatten the key into NodeArg* for better search.
      InlinedHashMap<const NodeArg*, GroupNode*> output_arg_to_grouped_node;
      for (auto& [output_args, grouped_node] : output_to_grouped_node) {
        for (const auto& output_arg : grouped_node.output_args) {
          std::cout << ">>>>> output_arg: " << output_arg->Name() << " is generated by group node" << std::endl;
          output_arg_to_grouped_node[output_arg] = &grouped_node;
        }
      }

      while (!to_visit.empty()) {
        const Node* current = to_visit.top();
        to_visit.pop();

        if (!current) continue;

        // if (enter) {
        std::cout << "enter " << current->Name() << std::endl;
        for (auto input_edge_it = current->InputEdgesBegin(); input_edge_it != current->InputEdgesEnd(); ++input_edge_it) {
          const NodeArg* output_arg = current->InputDefs()[input_edge_it->GetDstArgIndex()];
          if (already_ready.find(output_arg) == already_ready.end() && output_arg_to_grouped_node.find(output_arg) != output_arg_to_grouped_node.end()) {
            handle_group_node(&graph, output_arg, output_arg_to_grouped_node, node_orders, topo_order, already_ready);
          }
        }

        enter(current);

        for (const NodeArg* output_arg : current->OutputDefs()) {
          already_ready.insert(output_arg);
          std::cout << "update already_ready for output_arg: " << output_arg->Name() << std::endl;
        }
        // }

        for (auto output_edge_it = current->OutputEdgesBegin(); output_edge_it != current->OutputEdgesEnd(); ++output_edge_it) {
          const Node* node_it = &output_edge_it->GetNode();

          InlinedVector<const NodeArg*> left_args_generated_by_group_node;
          bool all_input_ready = true;
          bool all_not_ready_inputs_are_grouped = true;
          for (auto input_edge_it = node_it->InputEdgesBegin(); input_edge_it != node_it->InputEdgesEnd(); ++input_edge_it) {
            const NodeArg* input_arg = node_it->InputDefs()[input_edge_it->GetDstArgIndex()];

            if (already_ready.find(input_arg) == already_ready.end()) {
              std::cout << "input_arg: " << input_arg->Name() << " is not ready" << std::endl;
              all_input_ready = false;
              if (output_arg_to_grouped_node.find(input_arg) != output_arg_to_grouped_node.end()) {
                std::cout << "consider insert group node for " << input_arg->Name() << std::endl;
                left_args_generated_by_group_node.push_back(input_arg);
              } else {
                std::cout << "not-ready input_arg: " << input_arg->Name() << " is not grouped" << std::endl;
                all_not_ready_inputs_are_grouped = false;
                break;
              }
            }
          }

          if (all_input_ready) {
            std::cout << "push " << node_it->Name() << " into queue" << std::endl;
            to_visit.push(&*node_it);
          } else if (all_not_ready_inputs_are_grouped) {
            std::cout << "push " << node_it->Name() << " into queue (groupnode)" << std::endl;
            to_visit.push(&*node_it);
          }
        }

        topo_order.push_back(current->Index());
      }

      if (number_of_nodes != topo_order.size()) {
        ORT_THROW("Some nodes are not included in the topological sort, graph have a cycle. " + std::to_string(number_of_nodes) + " vs " + std::to_string(topo_order.size()));
      }
    };

    sort([&node_orders](const Node* n) -> void { node_orders.push_back(n->Index()); std::cout << "pengwa<<<<" << n->Name() << std::endl; }, PriorityNodeCompare(), node_orders);

    nodes_in_topological_order_with_priority_ = node_orders;
    ORT_ENFORCE(nodes_in_topological_order_with_priority_.size() == static_cast<size_t>(NumberOfNodes()), "Topological sort failed.", nodes_in_topological_order_with_priority_.size(), "!=", NumberOfNodes());
  } else {
    graph.KahnsTopologicalSort(
        [this](const Node* n) {
          nodes_in_topological_order_with_priority_.push_back(n->Index());
        },
        PriorityNodeCompare());
  }

  // // Tune the order a bit.
  // // If a node is used by a consumer node, but the execution order of the consumer node is later than the producer node,
  // // we should move the producer node to right before the consumer node. In this case, the producer node will only be executed
  // // when needed and the memory can be released earlier. We do this in reversed topological order to hanlde the single-input-single_output
  // // node chains.
  // InlinedVector<NodeIndex> node_in_reversed_order;
  // node_in_reversed_order.reserve(nodes_in_topological_order_with_priority.size());
  // for (auto it = nodes_in_topological_order_with_priority.rbegin(); it != nodes_in_topological_order_with_priority.rend(); ++it) {
  //   const Node* node = graph_->GetNode(*it);

  //   if (node->GetOutputEdgesCount() != 1) {
  //     // Don't need tune, just add it to the front of reversed_order.
  //     node_in_reversed_order.push_back(node->Index());
  //     continue;
  //   }

  //   // Handle the "High priority nodes" differently
  //   // So, it may break the computation order also when recompute is enabled.
  //   // But as ShapeInputMerge is introduced, there is much less chance to let recompute subgraph consumed by a normal Shape
  //   // or Size node. (TODO: pengwa): observe from real models and see if we need to handle this case.
  //   if (node->OpType() == "Shape" || node->OpType() == "Size") {
  //     node_in_reversed_order.push_back(node->Index());
  //     continue;
  //   }

  //   // If node is PythonOpGrad, and its attribute func_name string does not start with "onnxruntime.training.ortmodule._mem_efficient_grad_mgmt.ParamRetrievalFunction",
  //   // we skip to make sure the weight accumulation nodes are executed as early as possible (free buffer and unblock subsquent CPU work).
  //   if (node->OpType() == "PythonOpGrad") {
  //     const auto& attrs = node->GetAttributes();
  //     auto it = attrs.find("func_name");
  //     ORT_ENFORCE(it != attrs.end());
  //     if (it->second.s().find("onnxruntime.training.ortmodule._mem_efficient_grad_mgmt.ParamRetrievalFunction") != std::string::npos) {
  //       node_in_reversed_order.push_back(node->Index());
  //       continue;
  //     }
  //   }

  //   const Node* consumer = &(*(node->OutputNodesBegin()));
  //   // Insert the consumer node right after the producer node. (Remember the order is reversed here).
  //   auto it_consumer = std::find(node_in_reversed_order.begin(), node_in_reversed_order.end(), consumer->Index());
  //   ORT_ENFORCE(it_consumer != node_in_reversed_order.end());
  //   node_in_reversed_order.insert(it_consumer + 1, node->Index());  // Then node is inserted right after the consumer node.
  // }

  // nodes_in_topological_order_with_priority_.insert(
  //     nodes_in_topological_order_with_priority_.end(),
  //     node_in_reversed_order.rbegin(),
  //     node_in_reversed_order.rend());
#endif

  if (filter_info_) {
    // validate. if something is off here it's a bug in our code
    for (NodeIndex idx : filter_info->nodes) {
      ORT_ENFORCE(graph_->GetNode(idx) != nullptr, "IndexedSubGraph contains values not present in the Graph");
    }

    // create set of node indexes as we need quick lookups and don't care about the order
    filtered_node_indices_ = FilteredNodeSet(filter_info->nodes.cbegin(),
                                             filter_info->nodes.cend());

    const auto& metadef = filter_info->GetMetaDef();

    filtered_node_inputs_.reserve(metadef->inputs.size());
    filtered_node_inputs_including_initializers_.reserve(metadef->inputs.size());

    for (const auto& input : metadef->inputs) {
      const auto* nodearg = graph.GetNodeArg(input);
      ORT_ENFORCE(nodearg, "Mismatch between Graph and IndexedSubGraph. Input not found:", input);
      filtered_node_inputs_including_initializers_.push_back(nodearg);
      if (!graph.IsInitializedTensor(input)) {
        filtered_node_inputs_.push_back(nodearg);
      }
    }

    for (const auto& output : metadef->outputs) {
      const auto* nodearg = graph.GetNodeArg(output);
      ORT_ENFORCE(nodearg, "Mismatch between Graph and IndexedSubGraph. Output not found:", output);
      filtered_node_outputs_.push_back(nodearg);
    }

    // filter nodes in topo order to just the nodes in filter_info_
    auto orig_order = std::move(nodes_in_topological_order_);
    nodes_in_topological_order_.reserve(filter_info->nodes.size());
    std::copy_if(orig_order.cbegin(), orig_order.cend(), std::back_inserter(nodes_in_topological_order_),
                 [this](NodeIndex idx) { return filtered_node_indices_.count(idx) != 0; });

    // Filter the initializers also
    // Get the names of all the inputs and implicit inputs of all the nodes in this subgraph
    for (const auto node_idx : filtered_node_indices_) {
      const auto* node = GetNode(node_idx);
      ORT_ENFORCE(node, "Mismatch between Graph and IndexedSubGraph. Node not found: ", node_idx);
      const ONNX_NAMESPACE::TensorProto* tensor = nullptr;
      for (const auto* node_input : node->InputDefs()) {
        if (graph.GetInitializedTensor(node_input->Name(), tensor)) {
          filtered_initializers_.insert({node_input->Name(), tensor});
        }
      }

      // The implicit inputs for subgraphs (if any)
      for (const auto* node_input : node->ImplicitInputDefs()) {
        if (graph.GetInitializedTensor(node_input->Name(), tensor)) {
          filtered_initializers_.insert({node_input->Name(), tensor});
        }
      }
    }

#if !defined(ORT_MINIMAL_BUILD)
    auto orig_priority_order = std::move(nodes_in_topological_order_with_priority_);
    nodes_in_topological_order_with_priority_.reserve(filter_info->nodes.size());
    std::copy_if(orig_priority_order.cbegin(), orig_priority_order.cend(),
                 std::back_inserter(nodes_in_topological_order_with_priority_),
                 [this](NodeIndex idx) { return filtered_node_indices_.count(idx) != 0; });
#endif
  }
}

// Graph name.
const std::string&
GraphViewer::Name() const noexcept {
  return (filter_info_ == nullptr) ? graph_->Name()
                                   : filter_info_->GetMetaDef()->name;
}

const std::string& GraphViewer::Description() const noexcept {
  // filter_info_ doesn't have description so return 'name' instead of nothing
  // and to disambiguate between the full graph's description
  return (filter_info_ == nullptr) ? graph_->Description()
                                   : filter_info_->GetMetaDef()->name;
}

bool GraphViewer::GetInitializedTensor(const std::string& tensor_name,
                                       const ONNX_NAMESPACE::TensorProto*& value) const {
  value = nullptr;

  // if we are using filtered subgraph, the initializer has to be part of the subgraph
  if (filter_info_ != nullptr && filtered_initializers_.find(tensor_name) == filtered_initializers_.cend())
    return false;

  return graph_->GetInitializedTensor(tensor_name, value);
}

bool GraphViewer::CanOverrideInitializer() const noexcept {
  return graph_->CanOverrideInitializer();
}

// Graph inputs excluding initializers.
const std::vector<const NodeArg*>& GraphViewer::GetInputs() const noexcept {
  return (filter_info_ == nullptr) ? graph_->GetInputs()
                                   : filtered_node_inputs_;
}
// Graph inputs including initializers. Contains no nullptr values.
// This will match the number and order of inputs from the GraphProto.
const std::vector<const NodeArg*>& GraphViewer::GetInputsIncludingInitializers() const noexcept {
  return (filter_info_ == nullptr) ? graph_->GetInputsIncludingInitializers()
                                   : filtered_node_inputs_including_initializers_;
}

// Graph outputs. Should have no nullptr values.
const std::vector<const NodeArg*>& GraphViewer::GetOutputs() const noexcept {
  return (filter_info_ == nullptr) ? graph_->GetOutputs()
                                   : filtered_node_outputs_;
}

bool GraphViewer::NodeProducesGraphOutput(const Node& node) const {
  const auto& outputs = GetOutputs();
  auto end_outputs = outputs.cend();
  for (auto output_def : node.OutputDefs()) {
    if (std::find(outputs.cbegin(), end_outputs, output_def) != end_outputs) {
      return true;
    }
  }
  return false;
}

// Get graph value infos.
const std::unordered_set<const NodeArg*>& GraphViewer::GetValueInfo() const noexcept {
  return graph_->GetValueInfo();
}

// Get const Node given specific node index. May return nullptr if node as been freed.
const Node* GraphViewer::GetNode(NodeIndex node_index) const {
  if (filter_info_ && filtered_node_indices_.count(node_index) == 0) {
    return nullptr;
  }

  return graph_->GetNode(node_index);
}

const ConstGraphNodes& GraphViewer::Nodes() const noexcept {
  return graph_nodes_;
}

int GraphViewer::NumberOfNodes() const noexcept {
  return (filter_info_ == nullptr) ? graph_->NumberOfNodes()
                                   : gsl::narrow_cast<int>(filter_info_->nodes.size());
}

int GraphViewer::MaxNodeIndex() const noexcept {
  return graph_->MaxNodeIndex();
}

const std::vector<NodeIndex>& GraphViewer::GetNodesInTopologicalOrder(ExecutionOrder order) const {
  switch (order) {
    case ExecutionOrder::DEFAULT:
      return nodes_in_topological_order_;
#if !defined(ORT_MINIMAL_BUILD)
    case ExecutionOrder::PRIORITY_BASED:
      return nodes_in_topological_order_with_priority_;
#endif
    default:
      ORT_THROW("Invalid ExecutionOrder");
  }
}

const std::vector<NodeIndex>& GraphViewer::GetRootNodes() const {
  // TODO: See if we need to calculate the root_nodes_ of the filtered graph.
  // GetRootNodes is only used by parallel executor currently, and isn't relevant to the usage of a filtered graph.
  ORT_ENFORCE(filter_info_ == nullptr, "Not supported with filtered graph.");

  return root_nodes_;
}

const InitializedTensorSet& GraphViewer::GetAllInitializedTensors() const noexcept {
  return (filter_info_ == nullptr)
             ? graph_->GetAllInitializedTensors()
             : filtered_initializers_;
}

const NodeArg* GraphViewer::GetNodeArg(const std::string& name) const {
  return graph_->GetNodeArg(name);
}

bool GraphViewer::IsSubgraph() const {
  return graph_->IsSubgraph();
}

bool GraphViewer::IsConstantInitializer(const std::string& name, bool check_outer_scope) const {
  return GetConstantInitializer(name, check_outer_scope) != nullptr;
}

bool GraphViewer::IsInitializedTensor(const std::string& name) const {
  return graph_->IsInitializedTensor(name);
}

const ONNX_NAMESPACE::TensorProto* GraphViewer::GetConstantInitializer(const std::string& initializer_name,
                                                                       bool check_outer_scope) const {
  return graph_->GetConstantInitializer(initializer_name, check_outer_scope);
}

#if !defined(ORT_MINIMAL_BUILD)
const std::unordered_set<std::string>& GraphViewer::GetOuterScopeNodeArgNames() const noexcept {
  return graph_->GetOuterScopeNodeArgNames();
}
#endif

}  // namespace onnxruntime
