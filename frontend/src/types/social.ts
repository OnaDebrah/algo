export interface ActivityResponse {
    id: number;
    user_id: number;
    username: string;
    activity_type: string;
    content: string;
    metadata_json: Record<string, any> | null;
    created_at: string;
}
