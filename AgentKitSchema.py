from marshmallow import Schema, fields, EXCLUDE

class VariableSchema(Schema):
    class Meta:
        unknown = EXCLUDE
    
    name = fields.Str(allow_none=True)
    description = fields.Str(allow_none=True)
    data_type = fields.Str(data_key="type", allow_none=True) # type
    required = fields.Bool(allow_none=True)

class VectorStoreSchema(Schema):
    class Meta:
        unknown = EXCLUDE
    
    type = fields.Str(allow_none=True)
    embeddings_model = fields.Str(allow_none=True)
    store_path = fields.Str(allow_none=True)

class SessionSchema(Schema):
    class Meta:
        unknown = EXCLUDE
    
    session_id = fields.Str(allow_none=True)
    session_store = fields.Str(allow_none=True)

class ConversationSettingsSchema(Schema):
    class Meta:
        unknown = EXCLUDE
    
    history_size = fields.Int(allow_none=True)
    implementation = fields.Str(allow_none=True)
    session = fields.Nested(SessionSchema, allow_none=True)

class MemorySchema(Schema):
    class Meta:
        unknown = EXCLUDE
    
    type = fields.Str(allow_none=True)
    vector_store = fields.Nested(VectorStoreSchema, allow_none=True)
    enabled = fields.Bool(allow_none=True)
    conversation_settings = fields.Nested(ConversationSettingsSchema, allow_none=True)

class LLMConfigSchema(Schema):
    class Meta:
        unknown = EXCLUDE
    
    model_name = fields.Str(allow_none=True)
    temperature = fields.Float(allow_none=True)
    max_tokens = fields.Int(allow_none=True)
    top_p = fields.Float(allow_none=True)
    frequency_penalty = fields.Float(allow_none=True)
    presence_penalty = fields.Float(allow_none=True)

class ChainSchema(Schema):
    class Meta:
        unknown = EXCLUDE
    
    next_agent = fields.Str(allow_none=True)
    condition = fields.Str(allow_none=True)
    name = fields.Str(allow_none=True)
    description = fields.Str(allow_none=True)
    agents = fields.List(fields.Str(), allow_none=True)

class LoggingSchema(Schema):
    class Meta:
        unknown = EXCLUDE
    
    enabled = fields.Bool(allow_none=True)
    log_level = fields.Str(allow_none=True)
    log_file = fields.Str(allow_none=True)

class AgentSchema(Schema):
    class Meta:
        unknown = EXCLUDE
    
    name = fields.Str(allow_none=True)
    description = fields.Str(allow_none=True)
    tools = fields.List(fields.Str(), allow_none=True)
    llm_config = fields.Nested(LLMConfigSchema, allow_none=True)
    prompt = fields.List(fields.Dict(), allow_none=True)
    memory = fields.Nested(MemorySchema, allow_none=True)
    type = fields.Str(allow_none=True)
    chains = fields.List(fields.Nested(ChainSchema), allow_none=True)
    tool_categories = fields.List(fields.Str(), allow_none=True)
    logging = fields.Nested(LoggingSchema, allow_none=True)
    max_iterations = fields.Int(allow_none=True)

class ToolSchema(Schema):
    class Meta:
        unknown = EXCLUDE
    
    id = fields.Str(allow_none=True)
    name = fields.Str(allow_none=True)
    path = fields.Str(required=True)
    description = fields.Str(allow_none=True)
    allowed_agents = fields.List(fields.Str(), allow_none=True)
    category = fields.Str(allow_none=True)
    data_type = fields.Str(data_key="type", allow_none=True) # type
    connection_string = fields.Str(allow_none=True)
    query_template = fields.Str(allow_none=True)

class DocSchema(Schema):
    class Meta:
        unknown = EXCLUDE
    
    id = fields.Str(allow_none=True)
    path = fields.Str(allow_none=True)
    url = fields.Str(allow_none=True)

class ProjectSchema(Schema):
    class Meta:
        unknown = EXCLUDE
    
    name = fields.Str(allow_none=True)
    description = fields.Str(allow_none=True)
    root_dir = fields.Str(allow_none=True)

class PromptSchema(Schema):
    class Meta:
        unknown = EXCLUDE
    
    id = fields.Str(allow_none=True)
    path = fields.Str(allow_none=True)
    prompt = fields.Str(allow_none=True)
    variables = fields.List(fields.Nested(VariableSchema), allow_none=True)
    template = fields.Str(allow_none=True)

class StoreConfigSchema(Schema):
    class Meta:
        unknown = EXCLUDE
    
    host = fields.Str(allow_none=True)
    port = fields.Int(allow_none=True)
    user = fields.Str(allow_none=True)
    password = fields.Str(allow_none=True)

class StoreEntrySchema(Schema):
    class Meta:
        unknown = EXCLUDE
    
    id = fields.Str(allow_none=True)
    type = fields.Str(allow_none=True)
    config = fields.Nested(StoreConfigSchema, allow_none=True)

class AgentKitSchema(Schema):
    class Meta:
        unknown = EXCLUDE
    
    version = fields.Str(allow_none=True)
    project = fields.Nested(ProjectSchema, allow_none=True)
    docs = fields.List(fields.Nested(DocSchema), allow_none=True)
    tools = fields.List(fields.Nested(ToolSchema), allow_none=True)
    agents = fields.List(fields.Nested(AgentSchema), allow_none=True)
    prompts = fields.List(fields.Nested(PromptSchema), allow_none=True)
    store = fields.List(fields.Nested(StoreEntrySchema), allow_none=True)
    logging = fields.Nested(LoggingSchema, allow_none=True)
    memory = fields.Nested(MemorySchema, allow_none=True)
    chains = fields.List(fields.Nested(ChainSchema), allow_none=True)
