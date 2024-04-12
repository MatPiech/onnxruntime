// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/optimizer/cast_sce_loss_fusion.h"

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

Status CastSceLossFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                        const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (!node_ptr) continue;  // Node was removed.

    auto& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    bool is_internal_sce = graph_utils::IsSupportedOptypeVersionAndDomain(node, "SoftmaxCrossEntropyLossInternal", {1},
                                                                          kMSDomain);

    if (!is_internal_sce || !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders())) {
      continue;
    }

    Node* input_node = graph.GetNode(node.InputNodesBegin()->Index());

    if (!(graph_utils::IsSupportedOptypeVersionAndDomain(input_node, "Cast", {9, 13, 19}))) {
      continue;
    }

    // check the output of cast has only one consumer
    if (input_node->GetOutputEdgesCount() != 1) {
      continue;
    }

    if (input_node->MutableInputDefs()[0].TypeAsProto()->tensor_type().elem_type() == TensorProto_DataType_FLOAT16 &&
        input_node->MutableOutputDefs()[0].TypeAsProto()->tensor_type().elem_type() == TensorProto_DataType_FLOAT) {
      std::vector<GraphEdge> input_edges = GraphEdge::GetNodeInputEdges(node, 0);
      GraphEdge::RemoveGraphEdges(graph, input_edges);
      node.MutableInputDefs()[0] = input_node->MutableInputDefs()[0];
      MoveAllNodeInputEdges(graph, *input_node, node);
      graph.RemoveNode(input_node->Index());
      modified = true;
    }



  }

  return Status::OK();
}

}
