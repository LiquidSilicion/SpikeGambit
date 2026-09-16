// ============================================================================
// Module: spikegambit_top
// Description: Complete SpikeGambit SCNN accelerator
//              Integrates all layers: Conv1 -> Conv2 -> Conv3 -> Classifier
// ============================================================================

module spikegambit_top (
    input  wire clk,
    input  wire rst_n,
    
    // Input interface (board encoder)
    input  wire input_spike,
    input  wire [3:0] input_channel,   // 4 bits for up to 12 channels
    input  wire [2:0] input_x,         // 3 bits for 8x8 board
    input  wire [2:0] input_y,         // 3 bits for 8x8 board
    input  wire input_valid,
    
    // Output interface
    output wire [0:0] predicted_class, // 1 bit for 2 classes
    output wire output_valid,
    
    // Status
    output wire busy
);

    // Internal spike buses (Explicit widths to prevent padding warnings)
    wire conv1_spike;
    wire [4:0] conv1_channel;  // 5 bits for 32 channels
    wire [2:0] conv1_x, conv1_y;  // 3 bits for 8x8 board
    
    wire conv2_spike;
    wire [5:0] conv2_channel;  // 6 bits for 64 channels
    wire [1:0] conv2_x, conv2_y;  // 2 bits for 4x4 board
    
    wire conv3_spike;
    wire [6:0] conv3_channel;  // 7 bits for 128 channels
    wire [0:0] conv3_x, conv3_y;  // 1 bit for 2x2 board
    
    // Weight memory interfaces (widths match ADDR_WIDTH of each layer)
    wire [11:0] conv1_weight_addr;  // 12 bits for 4096 depth
    wire signed [15:0] conv1_weight_data;
    
    wire [14:0] conv2_weight_addr;  // 15 bits for 32768 depth
    wire signed [15:0] conv2_weight_data;
    
    wire [16:0] conv3_weight_addr;  // 17 bits for 131072 depth
    wire signed [15:0] conv3_weight_data;
    
    // Control signals
    wire conv1_busy, conv2_busy, conv3_busy;
    wire classifier_busy;
    
    assign busy = conv1_busy | conv2_busy | conv3_busy | classifier_busy;
    
    // ========================================================================
    // Layer 1: Conv1 (12 -> 32 channels, 3x3 kernel = 3,456 weights)
    // ========================================================================
    weight_memory #(
        .DATA_WIDTH(16),
        .ADDR_WIDTH(12),
        .DEPTH(4096),
        .FILE_NAME("/home/johan2/Documents/fpga/SpikeGambit/hw/rtl/conv1_weights.mem")
    ) conv1_weights (
        .clk(clk),
        .rst_n(rst_n),
        .read_addr(conv1_weight_addr),
        .read_data(conv1_weight_data),
        .read_en(1'b1)
    );
    
    conv_layer #(
        .IN_CHANNELS(12),
        .OUT_CHANNELS(32),
        .BOARD_SIZE(8),
        .DATA_WIDTH(16),
        .WEIGHT_ADDR_WIDTH(12)
    ) conv1 (
        .clk(clk),
        .rst_n(rst_n),
        .in_spike(input_spike),
        .in_channel(input_channel),
        .in_x(input_x),
        .in_y(input_y),
        .weight_addr(conv1_weight_addr),
        .weight_data(conv1_weight_data),
        .out_spike(conv1_spike),
        .out_channel(conv1_channel),
        .out_x(conv1_x),
        .out_y(conv1_y),
        .busy(conv1_busy)
    );
    
    // ========================================================================
    // Layer 2: Conv2 (32 -> 64 channels, 3x3 kernel = 18,432 weights)
    // ========================================================================
    weight_memory #(
        .DATA_WIDTH(16),
        .ADDR_WIDTH(15),      // Increased to address 32768
        .DEPTH(32768),
        .FILE_NAME("/home/johan2/Documents/fpga/SpikeGambit/hw/rtl/conv2_weights.mem")
    ) conv2_weights (
        .clk(clk),
        .rst_n(rst_n),
        .read_addr(conv2_weight_addr),
        .read_data(conv2_weight_data),
        .read_en(1'b1)
    );
    
    conv_layer #(
        .IN_CHANNELS(32),
        .OUT_CHANNELS(64),
        .BOARD_SIZE(4),
        .DATA_WIDTH(16),
        .WEIGHT_ADDR_WIDTH(15)
    ) conv2 (
        .clk(clk),
        .rst_n(rst_n),
        .in_spike(conv1_spike),
        .in_channel(conv1_channel),
        .in_x(conv1_x[2:1]),  // Pooling: divide by 2 (takes upper 2 bits)
        .in_y(conv1_y[2:1]),
        .weight_addr(conv2_weight_addr),
        .weight_data(conv2_weight_data),
        .out_spike(conv2_spike),
        .out_channel(conv2_channel),
        .out_x(conv2_x),
        .out_y(conv2_y),
        .busy(conv2_busy)
    );
    
    // ========================================================================
    // Layer 3: Conv3 (64 -> 128 channels, 3x3 kernel = 73,728 weights)
    // ========================================================================
    weight_memory #(
        .DATA_WIDTH(16),
        .ADDR_WIDTH(17),      // Increased to address 131072
        .DEPTH(131072),
        .FILE_NAME("/home/johan2/Documents/fpga/SpikeGambit/hw/rtl/conv3_weights.mem")
    ) conv3_weights (
        .clk(clk),
        .rst_n(rst_n),
        .read_addr(conv3_weight_addr),
        .read_data(conv3_weight_data),
        .read_en(1'b1)
    );
    
    conv_layer #(
        .IN_CHANNELS(64),
        .OUT_CHANNELS(128),
        .BOARD_SIZE(2),
        .DATA_WIDTH(16),
        .WEIGHT_ADDR_WIDTH(17)
    ) conv3 (
        .clk(clk),
        .rst_n(rst_n),
        .in_spike(conv2_spike),
        .in_channel(conv2_channel),
        .in_x(conv2_x[0]),  // 1 bit for 2x2 board
        .in_y(conv2_y[0]),
        .weight_addr(conv3_weight_addr),
        .weight_data(conv3_weight_data),
        .out_spike(conv3_spike),
        .out_channel(conv3_channel),
        .out_x(conv3_x),
        .out_y(conv3_y),
        .busy(conv3_busy)
    );
    
    // ========================================================================
    // Output Classifier
    // ========================================================================
    output_classifier #(
        .NUM_CLASSES(2),
        .TIME_STEPS(200),
        .DATA_WIDTH(16),
        .CLASS_ID_WIDTH(1)
    ) classifier (
        .clk(clk),
        .rst_n(rst_n),
        .spike_in(conv3_spike),
        .class_id(conv3_channel[0]),  // Simplified class ID mapping
        .input_valid(conv3_spike),
        .predicted_class(predicted_class),
        .output_valid(output_valid),
        .start(input_valid),
        .busy(classifier_busy)
    );

endmodule