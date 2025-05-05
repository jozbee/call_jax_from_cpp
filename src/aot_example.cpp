// call a simple xla script, preferably a couple of times
// we don't want to require all of the xla dependencies, only the runtime
//  shared library

#include <vector>
#include <iostream>
#include <fstream>

#include "artifacts/pjrt_c_api.h"
#include "artifacts/pjrt_c_api_cpu.h"


int main()
{
  // read compiled program
  const std::string input_binary = "artifacts/jax_example.binpb";
  std::ifstream input_file(input_binary, std::ios::binary);
  std::vector<char> buffer(
    // the extra parentheses are important
    (std::istreambuf_iterator<char>(input_file)),
    std::istreambuf_iterator<char>()
  );

  std::cout << "read binary\n";

  // api
  const PJRT_Api* api = GetPjrtApi();

  std::cout << "api\n";

  // client
  PJRT_Client_Create_Args client_create_args = {0};
  client_create_args.struct_size = sizeof(PJRT_Client_Create_Args);
  api->PJRT_Client_Create(&client_create_args);  // error?
  PJRT_Client* client = client_create_args.client;

  std::cout << "client\n";

  // load executable
  PJRT_Executable_DeserializeAndLoad_Args decereal_args = {0};
  decereal_args.struct_size = sizeof(PJRT_Executable_DeserializeAndLoad_Args);
  decereal_args.client = client;
  decereal_args.serialized_executable = buffer.data();
  decereal_args.serialized_executable_size = buffer.size();
  api->PJRT_Executable_DeserializeAndLoad(&decereal_args);
  PJRT_LoadedExecutable* loaded_executable = decereal_args.loaded_executable;

  std::cout << "load executable\n";

  // executable
  PJRT_LoadedExecutable_GetExecutable_Args executable_args = {0};
  executable_args.loaded_executable = loaded_executable;
  executable_args.struct_size = sizeof(
    PJRT_LoadedExecutable_GetExecutable_Args);
  api->PJRT_LoadedExecutable_GetExecutable(&executable_args);
  PJRT_Executable* executable = executable_args.executable;

  std::cout << "get executable\n";

  // outputs
  PJRT_Executable_NumOutputs_Args outputs_args = {0};
  outputs_args.executable = executable;
  api->PJRT_Executable_NumOutputs(&outputs_args);
  std::vector<PJRT_Buffer*> outputs(outputs_args.num_outputs);

  std::cout << "setup outputs";

  return 0;
}
