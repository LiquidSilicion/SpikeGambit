// ============================================================================
// Module: weight_memory
// Description: BRAM-based weight storage. Parameterized for different layers.
// ============================================================================

module weight_memory #(
    parameter DATA_WIDTH = 16,
    parameter ADDR_WIDTH = 12,
    parameter DEPTH = 4096,
    parameter FILE_NAME = "dummy.mem"
)(
    input  wire clk,
    // Removed rst_n since it's not used in BRAM
    
    // Read interface
    input  wire [ADDR_WIDTH-1:0] read_addr,
    output reg signed [DATA_WIDTH-1:0] read_data,
    input  wire read_en
);

    // BRAM storage
    reg signed [DATA_WIDTH-1:0] mem [0:DEPTH-1];
    
    // Initialize from .mem file (simulation only)
    initial begin
        $readmemh(FILE_NAME, mem);
    end
    
    // Synchronous read (BRAM behavior)
    always @(posedge clk) begin
        if (read_en) begin
            read_data <= mem[read_addr];
        end
    end

endmodule