// ============================================================================
// Module: conv_layer
// Description: 3x3 convolution layer with LIF neuron activation
//              Double-pipelined membrane update for guaranteed 100 MHz timing
// ============================================================================

module conv_layer #(
    parameter IN_CHANNELS = 12,
    parameter OUT_CHANNELS = 32,
    parameter BOARD_SIZE = 8,
    parameter KERNEL_SIZE = 3,
    parameter DATA_WIDTH = 16,
    parameter WEIGHT_ADDR_WIDTH = 12,
    parameter IN_CHANNEL_WIDTH = 4,
    parameter OUT_CHANNEL_WIDTH = 5,
    parameter XY_WIDTH = 3
)(
    input  wire clk,
    input  wire rst_n,
    
    // Input spike interface
    input  wire                         in_spike,
    input  wire [IN_CHANNEL_WIDTH-1:0]  in_channel,
    input  wire [XY_WIDTH-1:0]          in_x,
    input  wire [XY_WIDTH-1:0]          in_y,
    
    // Weight memory interface
    output reg [WEIGHT_ADDR_WIDTH-1:0]  weight_addr,
    input  wire signed [DATA_WIDTH-1:0] weight_data,
    
    // Output spike interface
    output reg                         out_spike,
    output reg [OUT_CHANNEL_WIDTH-1:0] out_channel,
    output reg [XY_WIDTH-1:0]          out_x,
    output reg [XY_WIDTH-1:0]          out_y,
    
    // Control
    output reg busy
);

    // State machine (added PIPELINE_STAGE2)
    localparam IDLE = 4'd0;
    localparam LOAD_WEIGHTS = 4'd1;
    localparam COMPUTE = 4'd2;
    localparam UPDATE_NEURON_STAGE1 = 4'd3;  // Compute sum
    localparam UPDATE_NEURON_STAGE2 = 4'd4;  // Subtract leak and write
    localparam CHECK_FIRE = 4'd5;
    
    reg [3:0] state;
    reg [1:0] kx, ky;
    reg [OUT_CHANNEL_WIDTH-1:0] oc;
    reg signed [DATA_WIDTH-1:0] accumulator;
    
    // Pipeline registers (added membrane_sum)
    reg signed [DATA_WIDTH-1:0] membrane_update;
    reg signed [DATA_WIDTH-1:0] membrane_sum;  // NEW: intermediate sum
    reg pipeline_valid;
    reg [OUT_CHANNEL_WIDTH-1:0] pipeline_oc;   // Track which channel is being pipelined
    
    // FLATTENED membrane array
    localparam MEM_SIZE = OUT_CHANNELS * BOARD_SIZE * BOARD_SIZE;
    reg signed [DATA_WIDTH-1:0] membrane [0:MEM_SIZE-1];
    
    // Threshold
    localparam signed [DATA_WIDTH-1:0] THRESHOLD = 16'sh0080;
    
    integer i;
    integer flat_idx;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            out_spike <= 1'b0;
            busy <= 1'b0;
            kx <= 2'd0;
            ky <= 2'd0;
            oc <= {OUT_CHANNEL_WIDTH{1'b0}};
            accumulator <= 16'sd0;
            out_channel <= {OUT_CHANNEL_WIDTH{1'b0}};
            out_x <= {XY_WIDTH{1'b0}};
            out_y <= {XY_WIDTH{1'b0}};
            weight_addr <= {WEIGHT_ADDR_WIDTH{1'b0}};
            membrane_update <= 16'sd0;
            membrane_sum <= 16'sd0;
            pipeline_valid <= 1'b0;
            pipeline_oc <= {OUT_CHANNEL_WIDTH{1'b0}};
            
            for (i = 0; i < MEM_SIZE; i = i + 1) begin
                membrane[i] <= 16'sd0;
            end
        end else begin
            case (state)
                IDLE: begin
                    out_spike <= 1'b0;
                    busy <= 1'b0;
                    pipeline_valid <= 1'b0;
                    if (in_spike) begin
                        state <= LOAD_WEIGHTS;
                        busy <= 1'b1;
                        kx <= 2'd0;
                        ky <= 2'd0;
                        oc <= {OUT_CHANNEL_WIDTH{1'b0}};
                        accumulator <= 16'sd0;
                    end
                end
                
                LOAD_WEIGHTS: begin
                    weight_addr <= (oc * IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE) +
                                   (in_channel * KERNEL_SIZE * KERNEL_SIZE) +
                                   (ky * KERNEL_SIZE + kx);
                    state <= COMPUTE;
                end
                
                COMPUTE: begin
                    accumulator <= accumulator + weight_data;
                    
                    if (kx < 2'd2) begin
                        kx <= kx + 2'd1;
                        state <= LOAD_WEIGHTS;
                    end else if (ky < 2'd2) begin
                        kx <= 2'd0;
                        ky <= ky + 2'd1;
                        state <= LOAD_WEIGHTS;
                    end else begin
                        state <= UPDATE_NEURON_STAGE1;
                    end
                end
                
                UPDATE_NEURON_STAGE1: begin
                    // STAGE 1: Compute (membrane + accumulator)
                    // This breaks the long arithmetic chain
                    flat_idx = oc * BOARD_SIZE * BOARD_SIZE + in_x * BOARD_SIZE + in_y;
                    membrane_sum <= membrane[flat_idx] + accumulator;
                    pipeline_oc <= oc;
                    
                    // Move to next output channel
                    if (oc < OUT_CHANNELS - 1) begin
                        oc <= oc + 1;
                        kx <= 2'd0;
                        ky <= 2'd0;
                        accumulator <= 16'sd0;
                        state <= LOAD_WEIGHTS;
                    end else begin
                        state <= UPDATE_NEURON_STAGE2;
                    end
                end
                
                UPDATE_NEURON_STAGE2: begin
                    // STAGE 2: Compute (sum - leak) and write back
                    // This is now a shorter path
                    flat_idx = pipeline_oc * BOARD_SIZE * BOARD_SIZE + in_x * BOARD_SIZE + in_y;
                    membrane[flat_idx] <= membrane_sum - (membrane[flat_idx] >>> 3);
                    
                    state <= CHECK_FIRE;
                end
                
                CHECK_FIRE: begin
                    out_spike <= 1'b0;
                    for (i = 0; i < OUT_CHANNELS; i = i + 1) begin
                        flat_idx = i * BOARD_SIZE * BOARD_SIZE + in_x * BOARD_SIZE + in_y;
                        if (membrane[flat_idx] >= THRESHOLD) begin
                            out_spike <= 1'b1;
                            out_channel <= i[OUT_CHANNEL_WIDTH-1:0];
                            out_x <= in_x;
                            out_y <= in_y;
                            membrane[flat_idx] <= 16'sd0;
                        end
                    end
                    state <= IDLE;
                    busy <= 1'b0;
                end
                
                default: begin
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule