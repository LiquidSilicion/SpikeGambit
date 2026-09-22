## ============================================================================
## SpikeGambit Constraints for ZedBoard (Zynq-7020)
## ============================================================================
## Pin assignments from ZedBoard Hardware User's Guide v2.2
## ============================================================================

## ============================================================================
## CLOCK - 100 MHz oscillator on Bank 13 (3.3V)
## Note: The Clocking Wizard IP automatically creates the clock constraint.
## We only need to assign the physical pin here.
## ============================================================================
set_property -dict { PACKAGE_PIN Y9 IOSTANDARD LVCMOS33 } [get_ports { clk_75mhz }];
## REMOVED: create_clock line - let Clocking Wizard handle it

## ============================================================================
## RESET - Button Left (BTNL) on Bank 34
## ============================================================================
set_property -dict { PACKAGE_PIN N15 IOSTANDARD LVCMOS25 } [get_ports { rst_n }];

## ============================================================================

set_property -dict { PACKAGE_PIN F22 IOSTANDARD LVCMOS25 } [get_ports { input_spike }];
set_property -dict { PACKAGE_PIN G22 IOSTANDARD LVCMOS25 } [get_ports { input_channel[0] }];
set_property -dict { PACKAGE_PIN H22 IOSTANDARD LVCMOS25 } [get_ports { input_channel[1] }];
set_property -dict { PACKAGE_PIN F21 IOSTANDARD LVCMOS25 } [get_ports { input_channel[2] }];
set_property -dict { PACKAGE_PIN H19 IOSTANDARD LVCMOS25 } [get_ports { input_channel[3] }];
set_property -dict { PACKAGE_PIN H18 IOSTANDARD LVCMOS25 } [get_ports { input_x[0] }];
set_property -dict { PACKAGE_PIN H17 IOSTANDARD LVCMOS25 } [get_ports { input_x[1] }];
set_property -dict { PACKAGE_PIN M15 IOSTANDARD LVCMOS25 } [get_ports { input_x[2] }];

## ============================================================================
## INPUT INTERFACE - Push Buttons on Bank 34
## ============================================================================
set_property -dict { PACKAGE_PIN T18 IOSTANDARD LVCMOS25 } [get_ports { input_y[0] }];
set_property -dict { PACKAGE_PIN R18 IOSTANDARD LVCMOS25 } [get_ports { input_y[1] }];
set_property -dict { PACKAGE_PIN R16 IOSTANDARD LVCMOS25 } [get_ports { input_y[2] }];
set_property -dict { PACKAGE_PIN P16 IOSTANDARD LVCMOS25 } [get_ports { input_valid }];

## ============================================================================
## OUTPUT INTERFACE - User LEDs on Bank 33 (3.3V)
## ============================================================================
set_property -dict { PACKAGE_PIN T22 IOSTANDARD LVCMOS33 } [get_ports { predicted_class[0] }];
set_property -dict { PACKAGE_PIN T21 IOSTANDARD LVCMOS33 } [get_ports { output_valid }];
set_property -dict { PACKAGE_PIN U22 IOSTANDARD LVCMOS33 } [get_ports { busy }];